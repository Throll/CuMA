import os
import json
import argparse
import random
from tqdm import tqdm
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from accelerate import Accelerator
from accelerate.utils import gather_object
from datasets import load_from_disk

from utils.func import set_seed
from utils.dataset.wvs_dataset import WVSDataset
from models.cuma_model import load_checkpoint
from models.mixlora_model import MixLoraModel
from models.hydralora_model import HydraLoraModel
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


class FewShotWVSDataset(WVSDataset):
    def __init__(self, train_dataset_path, k=3, **kwargs):
        super().__init__(**kwargs)
        self.k = k
        print(f"Loading few-shot training pool from {train_dataset_path}...")
        self.train_data = load_from_disk(train_dataset_path)
        
        # Group train data by country for better few-shot selection
        self.country_to_indices = {}
        for i in tqdm(range(len(self.train_data)), desc="Indexing training pool"):
            item = self.train_data[i]
            demo_info = parse_all_demographics(item['demographic'])
            country = demo_info.get('country', 'Unknown')
            if country not in self.country_to_indices:
                self.country_to_indices[country] = []
            self.country_to_indices[country].append(i)

    def get_collate_fn(self):
        # Get the base collate_fn from WVSDataset
        base_collate_fn = super().get_collate_fn()
        
        def collate_fn(batch):
            # For each item in batch, inject few-shot examples into 'messages'
            for item in batch:
                demo_info = parse_all_demographics(item['demographic'])
                country = demo_info.get('country', 'Unknown')
                
                # Sample k examples from the same country if possible
                if country in self.country_to_indices and len(self.country_to_indices[country]) >= self.k:
                    example_indices = random.sample(self.country_to_indices[country], self.k)
                else:
                    example_indices = random.sample(range(len(self.train_data)), self.k)
                
                examples = [self.train_data[i] for i in example_indices]
                
                # Original messages: [system, user, assistant]
                orig_messages = item['messages']
                system_msg = orig_messages[0]
                
                # Find the target user message (usually the last one before assistant)
                target_user_msg = None
                for msg in orig_messages:
                    if msg['role'] == 'user':
                        target_user_msg = msg
                        # In WVS dataset, there's usually only one user message
                
                new_messages = [system_msg]
                for ex in examples:
                    # ex['messages'] is [system, user, assistant]
                    new_messages.append(ex['messages'][1]) # User
                    new_messages.append(ex['messages'][2]) # Assistant
                
                new_messages.append(target_user_msg)
                item['messages'] = new_messages
            
            # Now call the base collate_fn which will handle apply_chat_template etc.
            return base_collate_fn(batch)
        
        return collate_fn


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
            if 'demographic_embed' in batch:
                demographic_embeds = batch['demographic_embed']
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
                    if isinstance(demo_embed, torch.Tensor):
                        demo_embed = demo_embed.to(dtype=torch_dtype)
                    else:
                        demo_embed = torch.tensor(demo_embed, dtype=torch_dtype)
                    demographic_embeds.append(demo_embed)
                demographic_embeds = torch.cat(demographic_embeds, dim=0).to(device=accelerator.device)
            elif hasattr(unwrapped_model, 'router_manager') and unwrapped_model.router_manager.routers[0].demographic_dim == 0:
                demographic_embeds = torch.zeros((input_ids.size(0), 0), dtype=torch_dtype, device=accelerator.device)
            else:
                raise ValueError("CuMA model requires demographic embeddings.")
            
            for router in unwrapped_model.router_manager.routers:
                router.set_demographic_embed(demographic_embeds)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False
        )
        
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
            for k, v in demo_infos[i].items():
                if k != 'country':
                    row[k.upper()] = v
            result_rows.append(row)
        
        if demographic_encoder:
            unwrapped_model = accelerator.unwrap_model(model)
            if hasattr(unwrapped_model, 'router_manager'):
                for router in unwrapped_model.router_manager.routers:
                    router.clear()
        
        accuracy = right_count / all_count if all_count > 0 else 0
        bar.set_postfix(acc=f"{accuracy:.4f}", cor=f"{right_count}/{all_count}")
    
    bar.close()
    
    all_rows = gather_object(result_rows)
    if isinstance(all_rows, list) and len(all_rows) > 0 and isinstance(all_rows[0], list):
        flat_rows = []
        for r in all_rows:
            flat_rows.extend(r)
        all_rows = flat_rows
        
    total_correct = sum(1 for r in all_rows if r.get('IS_CORRECT'))
    total_count = len(all_rows)
    final_accuracy = total_correct / total_count if total_count > 0 else 0
    
    return final_accuracy, total_correct, total_count, all_rows


def compute_metrics(df):
    from sklearn.metrics import f1_score, accuracy_score
    y_true = df['GROUND_TRUTH'].astype(str).tolist()
    y_pred = df['PREDICTED_LETTER'].astype(str).tolist()
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    
    countries = df['COUNTRY'].unique()
    country_accuracies = {}
    for country in countries:
        if country == 'Unknown': continue
        country_df = df[df['COUNTRY'] == country]
        if len(country_df) == 0: continue
        acc = accuracy_score(country_df['GROUND_TRUTH'], country_df['PREDICTED_LETTER'])
        country_accuracies[country] = acc
    
    macro_acc_country = np.mean(list(country_accuracies.values())) if country_accuracies else 0.0
    
    dimensions = ['COUNTRY', 'AGE_GROUP', 'GENDER', 'EDUCATION', 'MARITAL_STATUS', 'RELIGION', 'ETHNICITY', 'EMPLOYMENT']
    sigmas = []
    for dim in dimensions:
        if dim not in df.columns: continue
        subgroup_accs = []
        subgroups = df[dim].unique()
        for sg in subgroups:
            if sg == 'Unknown' or pd.isna(sg): continue
            sg_df = df[df[dim] == sg]
            if len(sg_df) < 5: continue
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
    emd_dict = emd_evaluate(csv_path, output_file=None, sep=sep)
    if not emd_dict:
        raise ValueError("EMD evaluation returned empty results")
    emd_values = list(emd_dict.values())
    emd_array = np.array(emd_values)
    percentages = [(emd_array <= th).sum() * 100.0 / len(emd_array) for th in thresholds]
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
    tokenizer.padding_side = 'right'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if accelerator.is_main_process:
        print(f"Loading base model from {args.model}")
    torch_dtype = torch.bfloat16 if args.bf16 else torch.float32
    
    # Load evaluation dataset
    if accelerator.is_main_process:
        print(f"Loading dataset from {args.dataset}")
    
    eval_dataset = FewShotWVSDataset(
        train_dataset_path=args.train_dataset,
        k=args.few_shot,
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

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch_dtype,
        device_map={"": device}
    )
    
    adapter_type = args.adapter_type.lower()
    if adapter_type == 'cuma':
        embedding_model_to_load = args.embedding_model if not has_precomputed_embeds else None
        model, demographic_encoder, _ = load_checkpoint(
            base_model=base_model,
            adapter_path=args.peft_model_path,
            embedding_model_name=embedding_model_to_load,
            device=device
        )            
    elif adapter_type == 'mixlora':
        model = MixLoraModel.from_pretrained(base_model, args.peft_model_path)
        demographic_encoder = None
    elif adapter_type == 'hydralora':
        model = HydraLoraModel.from_pretrained(base_model, args.peft_model_path)
        demographic_encoder = None
    elif adapter_type in ['lora', 'prefix']:
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
    
    model, demographic_encoder, eval_dataloader = accelerator.prepare(
        model, demographic_encoder, eval_dataloader
    )
    
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
        df = pd.DataFrame(rows)
        extra_metrics = compute_metrics(df)
        
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            df.to_csv(os.path.join(args.output_dir, 'results.csv'), sep='\t', index=False)
            thresholds = np.arange(0.0, args.auc_threshold_max + 1e-8, args.auc_threshold_step)
            metrics = compute_auc_emd(os.path.join(args.output_dir, 'results.csv'), thresholds=thresholds, sep='\t')
            
            print(f"\nACC: {accuracy*100:.2f}% | Macro-F1: {extra_metrics['macro_f1']*100:.2f}% | EMD Mean: {metrics['emd_mean']*100:.2f}%")
            
            summary = {
                "accuracy": accuracy,
                "sigma_inequity": extra_metrics['sigma_inequity'],
                "macro_f1": extra_metrics['macro_f1'],
                "macro_acc_country": extra_metrics['macro_acc_country'],
                "correct": right_count,
                "total": all_count,
                "emd_mean": metrics['emd_mean'],
                "emd_median": metrics['emd_median'],
                "question_count": metrics['question_count'],
                "country_accuracies": extra_metrics['country_accuracies']
            }
            with open(os.path.join(args.output_dir, 'evaluation_summary.json'), 'w') as f:
                json.dump(summary, f, indent=2)

    # Clean up
    del model
    if demographic_encoder:
        del demographic_encoder
    import gc
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Few-Shot Persona Prompting')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--embedding_model', type=str, default=None)
    parser.add_argument('--adapter_type', type=str, default='base', choices=['base', 'lora', 'cuma', 'prefix', 'mixlora', 'hydralora'])
    parser.add_argument('--peft_model_path', type=str, default="")
    parser.add_argument('--dataset', type=str, required=True, help='Test dataset path')
    parser.add_argument('--train_dataset', type=str, required=True, help='Train dataset path for few-shot examples')
    parser.add_argument('--few_shot', type=int, default=3)
    parser.add_argument('--eval_batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--bf16', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='')
    parser.add_argument('--auc_threshold_max', type=float, default=0.6)
    parser.add_argument('--auc_threshold_step', type=float, default=0.05)
    parser.add_argument("--use_demographic_system_prompt", action="store_true")
    
    args = parser.parse_args()
    main(args)
