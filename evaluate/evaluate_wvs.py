import os
import json
import argparse
from tqdm import tqdm
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from accelerate import Accelerator
from accelerate.utils import gather_object

from utils.func import set_seed
from utils.dataset import WVSDataset
from models.cuma_model import load_checkpoint
# from models.mixlora_model import MixLoraModel
# from models.hydralora_model import HydraLoraModel
from WorldValuesBench.evaluation.evaluate import evaluate as emd_evaluate


def format_demographics_dict(profile: dict) -> str:
    """Format WVS demographic dictionary into a string."""
    return (
        f"You are a person from {profile.get('country', 'Unknown')}, "
        f"age group: {profile.get('age_group', 'Unknown')}, "
        f"gender: {profile.get('gender', 'Unknown')}, "
        f"education: {profile.get('education', 'Unknown')}, "
        f"marital status: {profile.get('marital_status', 'Unknown')}, "
        f"religion: {profile.get('religion', 'Unknown')}, "
        f"ethnicity: {profile.get('ethnicity', 'Unknown')}, "
        f"employment: {profile.get('employment', 'Unknown')}."
    )


def parse_all_demographics(demo):
    """Parse demographic string or dict into a dictionary of fields."""
    if isinstance(demo, dict):
        return {k: str(v) for k, v in demo.items()}
    
    res = {}
    if not isinstance(demo, str):
        return res
        
    try:
        if "You are a person from " in demo:
            parts = demo.split("You are a person from ")
            rest = parts[1]
            country_part = rest.split(", age group: ")[0]
            res['country'] = country_part
            
            fields = ["age group", "gender", "education", "marital status", "religion", "ethnicity", "employment"]
            for i, field in enumerate(fields):
                search_str = f", {field}: "
                if search_str in rest:
                    val_part = rest.split(search_str)[1]
                    # Find next field or end
                    next_pos = -1
                    for next_field in fields[i+1:]:
                        next_search = f", {next_field}: "
                        if next_search in val_part:
                            next_pos = val_part.find(next_search)
                            break
                    
                    if next_pos != -1:
                        val = val_part[:next_pos]
                    else:
                        val = val_part.rstrip('.')
                    res[field.replace(" ", "_")] = val
    except:
        pass
    return res


@torch.no_grad()
def evaluate_wvs(model, demographic_encoder, eval_dataloader, tokenizer, accelerator, torch_dtype, adapter_type="base"):
    """Evaluate base / standard LoRA / CuMA models on multiple choice questions"""
    
    model.eval()
    if demographic_encoder:
        demographic_encoder.eval()
    
    right_count = 0
    all_count = 0
    result_rows = []
    
    # Only show progress bar on main process
    bar = tqdm(eval_dataloader, desc="Evaluating", dynamic_ncols=True, disable=not accelerator.is_local_main_process)
    
    for batch in bar:
        # Accelerate handles device placement
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        targets = batch['target']
        letter_choices_batch = batch['letter_choices']
        numeric_choices_batch = batch['numeric_choices']
        question_ids = batch.get('question_id', [-1] * len(targets))
        participant_ids = batch.get('participant_id', [-1] * len(targets))
        
        # Extract demographic info
        demo_infos = []
        if 'demographic' in batch:
            for profile in batch['demographic']:
                demo_infos.append(parse_all_demographics(profile))
        else:
            demo_infos = [{}] * len(targets)
        
        countries = [d.get('country', 'Unknown') for d in demo_infos]
        
        unwrapped_model = accelerator.unwrap_model(model)
        # CuMA specific routing requires demographic embeddings
        is_cuma = adapter_type == "cuma"

        if is_cuma:
            # Prefer pre-computed embeddings if present in dataset to avoid
            # re-encoding and potential encoder/model mismatch.
            if 'demographic_embed' in batch:
                demographic_embeds = batch['demographic_embed']
                # ensure tensor
                if not isinstance(demographic_embeds, torch.Tensor):
                    demographic_embeds = torch.tensor(demographic_embeds, dtype=torch_dtype)
                demographic_embeds = demographic_embeds.to(device=accelerator.device)
            elif demographic_encoder:
                demographic_embeds = []
                for profile in batch['demographic']:
                    if isinstance(profile, dict):
                        profile_text = format_demographics_dict(profile)
                    else:
                        profile_text = profile
                    demo_embed = demographic_encoder(profile_text)
                    # demographic_encoder may return cpu tensor; ensure correct dtype
                    if isinstance(demo_embed, torch.Tensor):
                        demo_embed = demo_embed.to(dtype=torch_dtype)
                    else:
                        demo_embed = torch.tensor(demo_embed, dtype=torch_dtype)
                    demographic_embeds.append(demo_embed)
                # Use cat instead of stack to handle [1, D] -> [B, D]
                demographic_embeds = torch.cat(demographic_embeds, dim=0).to(device=accelerator.device)
            elif hasattr(unwrapped_model, 'router_manager') and unwrapped_model.router_manager.routers[0].demographic_dim == 0:
                # Handle ablation case where demographic_embed_dim is 0
                demographic_embeds = torch.zeros((input_ids.size(0), 0), dtype=torch_dtype, device=accelerator.device)
            else:
                raise ValueError("CuMA model requires demographic embeddings, but none provided in batch and no encoder available.")
            
            # Access underlying model to set router embeddings
            for router in unwrapped_model.router_manager.routers:
                router.set_demographic_embed(demographic_embeds)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False
        )
        
        # last_logits = outputs.logits[:, -1, :]
        batch_size = input_ids.size(0)
        last_index = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(batch_size, device=accelerator.device)
        last_logits = outputs.logits[batch_indices, last_index]
                
        for i, (target, letter_choices, numeric_choices, qid, pid) in enumerate(
            zip(targets, letter_choices_batch, numeric_choices_batch, question_ids, participant_ids)
        ):
            if target is None:
                raise ValueError(f"Sample {i} has invalid target: {target}")
            ground_truth = target.strip()
            
            choice_token_ids = []
            for letter in letter_choices:
                tokens = tokenizer.encode(letter, add_special_tokens=False)
                if not tokens or len(tokens) != 1:
                    print(f"Error tokenizing choice '{letter}' for sample {i}: tokens={tokens}")
                    raise
                choice_token_ids.append(tokens[0])
            
            choice_index = torch.tensor(choice_token_ids, device=accelerator.device)
            sample_logits = last_logits[i, choice_index]
            
            prediction_idx = torch.argmax(sample_logits).item()
            predicted_letter = letter_choices[prediction_idx]
            
            score = numeric_choices[prediction_idx]
            
            # Check correctness
            is_correct = (predicted_letter == ground_truth)
            if is_correct:
                right_count += 1
            all_count += 1
            
            row = {
                "QUESTION_ID": qid,
                "PARTICIPANT_ID": pid,
                "SCORE": score,
                "IS_CORRECT": is_correct,
                "PREDICTED_LETTER": predicted_letter,
                "GROUND_TRUTH": ground_truth,
                "COUNTRY": countries[i]
            }
            # Add other demographic fields for σINEQUITY
            for k, v in demo_infos[i].items():
                if k != 'country':
                    row[k.upper()] = v
            result_rows.append(row)
        
        if demographic_encoder:
            unwrapped_model = accelerator.unwrap_model(model)
            if hasattr(unwrapped_model, 'router_manager'):
                for router in unwrapped_model.router_manager.routers:
                    router.clear()
        
        # Update progress bar
        accuracy = right_count / all_count if all_count > 0 else 0
        bar.set_postfix(acc=f"{accuracy:.4f}", cor=f"{right_count}/{all_count}")
    
    bar.close()
    
    # Gather results from all processes
    all_rows = gather_object(result_rows)
    
    # Flatten list of lists
    if isinstance(all_rows, list) and len(all_rows) > 0 and isinstance(all_rows[0], list):
        flat_rows = []
        for r in all_rows:
            flat_rows.extend(r)
        all_rows = flat_rows
        
    # Recompute metrics on gathered data
    total_correct = sum(1 for r in all_rows if r.get('IS_CORRECT'))
    total_count = len(all_rows)
    final_accuracy = total_correct / total_count if total_count > 0 else 0
    
    return final_accuracy, total_correct, total_count, all_rows


def compute_metrics(df):
    """Compute Macro-F1, Country-level Macro-Accuracy and σINEQUITY"""
    from sklearn.metrics import f1_score, accuracy_score
    
    # 1. Macro-F1
    y_true = df['GROUND_TRUTH'].astype(str).tolist()
    y_pred = df['PREDICTED_LETTER'].astype(str).tolist()
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    
    # 2. Country-level Macro-Accuracy
    countries = df['COUNTRY'].unique()
    country_accuracies = {}
    for country in countries:
        if country == 'Unknown': continue
        country_df = df[df['COUNTRY'] == country]
        if len(country_df) == 0: continue
        acc = accuracy_score(country_df['GROUND_TRUTH'], country_df['PREDICTED_LETTER'])
        country_accuracies[country] = acc
    
    macro_acc_country = np.mean(list(country_accuracies.values())) if country_accuracies else 0.0
    
    # 3. σINEQUITY (Value Equity Index)
    # Dimensions to consider (based on available fields)
    dimensions = ['COUNTRY', 'AGE_GROUP', 'GENDER', 'EDUCATION', 'MARITAL_STATUS', 'RELIGION', 'ETHNICITY', 'EMPLOYMENT']
    sigmas = []
    for dim in dimensions:
        if dim not in df.columns: continue
        subgroup_accs = []
        subgroups = df[dim].unique()
        for sg in subgroups:
            if sg == 'Unknown' or pd.isna(sg): continue
            sg_df = df[df[dim] == sg]
            if len(sg_df) < 5: continue # Skip very small subgroups for stability
            acc = accuracy_score(sg_df['GROUND_TRUTH'], sg_df['PREDICTED_LETTER'])
            subgroup_accs.append(acc)
        
        if len(subgroup_accs) > 1:
            sigmas.append(np.std(subgroup_accs))
            
    sigma_inequity = np.mean(sigmas) if sigmas else 0.0
        
    return {
        "macro_f1": float(macro_f1),
        "macro_acc_country": float(macro_acc_country),
        "sigma_inequity": float(sigma_inequity),
        "country_accuracies": country_accuracies
    }

def compute_auc_emd(csv_path: str, thresholds, sep: str = '\t'):
    """Compute per-question EMD and AUC from prediction CSV"""
    import numpy as np
    
    emd_dict = emd_evaluate(csv_path, output_file=None, sep=sep)
    if not emd_dict:
        raise ValueError("EMD evaluation returned empty results")

    emd_values = list(emd_dict.values())
    emd_array = np.array(emd_values)
    
    # Calculate percentage under each threshold
    percentages = [(emd_array <= th).sum() * 100.0 / len(emd_array) for th in thresholds]
    
    # Trapezoidal integration for AUC
    if hasattr(np, 'trapezoid'):
        auc = np.trapezoid(percentages, thresholds)
    else:
        auc = np.trapz(percentages, thresholds)
    
    return {
        "emd_mean": float(emd_array.mean()),
        "emd_median": float(np.median(emd_array)),
        "auc": float(auc),
        "question_count": len(emd_values),
        "thresholds": list(thresholds)
    }


def main(args):
    accelerator = Accelerator()
    device = accelerator.device
    set_seed(args.seed)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = 'right' # Ensure right padding for consistent indexing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if accelerator.is_main_process:
        print(f"Loading base model from {args.model}")
    torch_dtype = torch.bfloat16 if args.bf16 else torch.float32
    
    # Load evaluation dataset first to check for pre-computed embeddings
    if accelerator.is_main_process:
        print(f"Loading dataset from {args.dataset}")
    eval_dataset = WVSDataset(
        dataset_path=args.dataset,
        tokenizer=tokenizer,
        model_name_or_path=args.model,
        include_demographic=True,
        use_demographic_system_prompt=args.use_demographic_system_prompt,
        is_eval=True
    )
    
    has_precomputed_embeds = False
    if len(eval_dataset.data) > 0 and 'demographic_embed' in eval_dataset.data[0]:
        has_precomputed_embeds = True
        if accelerator.is_main_process:
            print("Detected pre-computed demographic embeddings in dataset.")

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch_dtype,
        device_map={"": device} # Let accelerate handle DDP, but here we force to local device
    )
    
    adapter_type = args.adapter_type.lower()
    if adapter_type == 'cuma':
        if not args.peft_model_path:
            raise ValueError("Cultural adapter selected but --peft_model_path is empty")
        
        # If we have pre-computed embeds, we don't strictly need the embedding_model
        if not args.embedding_model and not has_precomputed_embeds:
            raise ValueError("Cultural adapter selected but --embedding_model is empty and no pre-computed embeddings found in dataset")
        
        if accelerator.is_main_process:
            print(f"Loading CuMA from {args.peft_model_path}")
        
        # Only load encoder if we don't have pre-computed embeddings
        embedding_model_to_load = args.embedding_model if not has_precomputed_embeds else None
        
        model, demographic_encoder, _ = load_checkpoint(
            base_model=base_model,
            adapter_path=args.peft_model_path,
            embedding_model_name=embedding_model_to_load,
            device=device
        )            
    # elif adapter_type == 'mixlora':
    #     if not args.peft_model_path:
    #         raise ValueError("MixLoRA selected but --peft_model_path is empty")
    #     if accelerator.is_main_process:
    #         print(f"Loading MixLoRA from {args.peft_model_path}")
    #     model = MixLoraModel.from_pretrained(base_model, args.peft_model_path)
    #     demographic_encoder = None
    # elif adapter_type == 'hydralora':
    #     if not args.peft_model_path:
    #         raise ValueError("HydraLoRA selected but --peft_model_path is empty")
    #     if accelerator.is_main_process:
    #         print(f"Loading HydraLoRA from {args.peft_model_path}")
    #     model = HydraLoraModel.from_pretrained(base_model, args.peft_model_path)
    #     demographic_encoder = None
    elif adapter_type in ['lora', 'prefix']:
        if not args.peft_model_path:
            raise ValueError(f"{adapter_type.upper()} selected but --peft_model_path is empty")
        if accelerator.is_main_process:
            print(f"Loading {adapter_type.upper()} adapter from {args.peft_model_path}")
        model = PeftModel.from_pretrained(base_model, args.peft_model_path)
        demographic_encoder = None
    else:
        model = base_model
        demographic_encoder = None
    
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=eval_dataset.get_collate_fn(),
        num_workers=args.num_workers
    )
    
    # Prepare with accelerator
    model, demographic_encoder, eval_dataloader = accelerator.prepare(
        model, demographic_encoder, eval_dataloader
    )
    
    # Run evaluation
    if accelerator.is_main_process:
        print("Starting evaluation...")
    
    accuracy, right_count, all_count, rows = evaluate_wvs( 
        model=model,
        demographic_encoder=demographic_encoder,
        eval_dataloader=eval_dataloader,
        tokenizer=tokenizer,
        accelerator=accelerator,
        torch_dtype=torch_dtype,
        adapter_type=args.adapter_type
    )
    
    if accelerator.is_main_process:
        print(f"\n{'='*50}")
        print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)  Correct: {right_count}/{all_count}")
        print(f"{'='*50}\n")
        
        # Save results
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            
            import pandas as pd
            df = pd.DataFrame(rows)
            df.to_csv(os.path.join(args.output_dir, 'results.csv'), sep='\t', index=False)
            print(f"Saved predictions to {os.path.join(args.output_dir, 'results.csv')}")
            
            # Compute EMD & AUC
            import numpy as np
            thresholds = np.arange(0.0, args.auc_threshold_max + 1e-8, args.auc_threshold_step)
            metrics = compute_auc_emd(os.path.join(args.output_dir, 'results.csv'), thresholds=thresholds, sep='\t')
            
            # Compute Macro-F1 and Country-level Macro-Acc
            extra_metrics = compute_metrics(df)
            
            print(
                f"\nACC: {accuracy*100:.2f}% | Macro-F1: {extra_metrics['macro_f1']*100:.2f}% | EMD Mean: {metrics['emd_mean']*100:.2f}%"
            )
            
            os.makedirs(args.output_dir, exist_ok=True)
            summary = {
                "accuracy": accuracy,
                "sigma_inequity": extra_metrics['sigma_inequity'],
                "macro_f1": extra_metrics['macro_f1'],
                "macro_acc_country": extra_metrics['macro_acc_country'],
                "correct": right_count,
                "total": all_count,
                "emd_mean": metrics['emd_mean'],
                "emd_median": metrics['emd_median'],
                # "auc": metrics['auc'],
                "question_count": metrics['question_count'],
                "country_accuracies": extra_metrics['country_accuracies']
            }
            with open(os.path.join(args.output_dir, 'evaluation_summary.json'), 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"Saved summary to {args.output_dir}/evaluation_summary.json")

    # Clean up
    del model
    if demographic_encoder:
        del demographic_encoder
    import gc
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate CuMA')
    
    parser.add_argument('--model', type=str, required=True, help='Base model path')
    parser.add_argument('--embedding_model', type=str, default=None, help='Embedding model path')
    parser.add_argument('--adapter_type', type=str, default='base', choices=['base', 'lora', 'cuma', 'prefix', 'mixlora', 'hydralora'])
    parser.add_argument('--peft_model_path', type=str, default="", help='Adapter path')
    parser.add_argument('--dataset', type=str, required=True, help='Eval dataset path')
    parser.add_argument('--eval_batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--gpu', type=int, default=-1, help='GPU id, -1 for auto')
    parser.add_argument('--bf16', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='', help='Output dir')

    parser.add_argument('--auc_threshold_max', type=float, default=0.6)
    parser.add_argument('--auc_threshold_step', type=float, default=0.05)
    parser.add_argument("--use_demographic_system_prompt", action="store_true", help="Use demographic info as system prompt")
    
    args = parser.parse_args()
    main(args)