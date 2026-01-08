import os
import sys

# --- Configuration Switch ---

BASE_OUTPUT_DIR = "./data"
WVS_FULL_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, "wvs_sft")
TEMP_PROCESSING_DIR = None # Use default system temp

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import gzip
import pandas as pd
import numpy as np
import multiprocessing
import tempfile
import shutil
import torch
from typing import Dict, List, Optional
from datasets import Dataset, load_from_disk, load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from models.demographic_encoder import DemographicEncoder

# Configuration
# WVS Paths (Large dataset, store in file storage)
WVS_RAW_DIR = "./WorldValuesBench/WorldValuesBench"
WVS_CODEBOOK_PATH = "./WorldValuesBench/dataset_construction/codebook.json"
# WVS_FULL_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, "wvs_sft")
WVS_SAMPLED_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, "wvs_sampled_sft")

# Facebook Paths
FACEBOOK_INPUT_PATH = "../MyDatasets/facebook/community-alignment-dataset/benchmark/filtered"
FACEBOOK_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, "facebook_dpo")
FACEBOOK_OUTPUT_FILE = os.path.join(FACEBOOK_OUTPUT_DIR, "train.jsonl")

# PRISM Paths
PRISM_CONVERSATIONS_PATH = "../MyDatasets/HannahRoseKirk/prism-alignment/conversations/train"
PRISM_SURVEY_PATH = "../MyDatasets/HannahRoseKirk/prism-alignment/survey/train"
PRISM_SFT_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, "prism_sft")
PRISM_DPO_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, "prism_dpo")

# Model for Tokenizer (needed for Chat Template)
MODEL_NAME = "../MyModels/Qwen3-8B"

# --- Helper Functions ---

def add_demographic_embeddings(dataset, model_name="../MyModels/Qwen3-Embedding-0.6B", batch_size=256):
    print(f"Loading DemographicEncoder from {model_name}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = DemographicEncoder(model_name=model_name)
    encoder.to(device)
    encoder.eval()
    
    # Collect all unique demographics to save computation
    print("Collecting unique demographics...")
    unique_demos = set()
    
    # Handle DatasetDict or Dataset
    if isinstance(dataset, dict): # DatasetDict
        for split in dataset.keys():
            unique_demos.update(dataset[split]['demographic'])
    else:
        unique_demos.update(dataset['demographic'])
    
    unique_demos_list = list(unique_demos)
    print(f"Found {len(unique_demos_list)} unique demographic profiles.")
    
    # Compute embeddings for unique demographics
    demo_to_embed = {}
    print("Encoding demographics...")
    
    with torch.no_grad():
        for i in tqdm(range(0, len(unique_demos_list), batch_size)):
            batch_texts = unique_demos_list[i:i+batch_size]
            embeddings = encoder(batch_texts)
            embeddings = embeddings.cpu().numpy()
            
            for text, embed in zip(batch_texts, embeddings):
                demo_to_embed[text] = embed
                
    # Map back to dataset
    def add_embedding_batched(examples):
        return {'demographic_embed': [demo_to_embed[d] for d in examples['demographic']]}

    print("Mapping embeddings to dataset...")
    # Use batched=True for better performance and storage efficiency
    encoded_dataset = dataset.map(
        add_embedding_batched, 
        batched=True, 
        desc="Adding embeddings",
        num_proc=8  # Use parallel processing
    )
    
    return encoded_dataset

def load_wvs_raw_data(mode):
    """Load codebook, demographic data, and value data from TSV."""
    print(f"Loading WVS raw data from {WVS_RAW_DIR} ({mode} split)...")
    with open(WVS_CODEBOOK_PATH, 'r', encoding='utf-8') as f:
        codebook = json.load(f)
    
    demographic_df = pd.read_csv(
        f'{WVS_RAW_DIR}/{mode}/{mode}_demographic_qa.tsv', 
        sep='\t', 
        index_col='D_INTERVIEW'
    )
    
    value_df = pd.read_csv(
        f'{WVS_RAW_DIR}/{mode}/{mode}_value_qa.tsv',
        sep='\t',
        index_col='D_INTERVIEW'
    )
    return codebook, demographic_df, value_df

def extract_demographic_profile(demo_data) -> Dict:
    """Extract demographic profile as dictionary."""
    # demo_data can be a pandas Series or a dict
    country = demo_data.get('B_COUNTRY', 'Unknown')
    age_group = demo_data.get('X003R', 'Unknown')
    if isinstance(age_group, str) and age_group.endswith('years'):
        age_group = age_group.replace(' years', ' years old')
    
    return {
        'country': str(country),
        'age_group': str(age_group),
        'gender': str(demo_data.get('Q260', 'Unknown')),
        'education': str(demo_data.get('Q275', 'Unknown')),
        'marital_status': str(demo_data.get('Q273', 'Unknown')),
        'religion': str(demo_data.get('Q289', 'Unknown')),
        'ethnicity': str(demo_data.get('Q290', 'Unknown')),
        'employment': str(demo_data.get('Q279', 'Unknown'))
    }

def format_demographics_dict(profile: dict) -> str:
    """Format WVS demographic dictionary into a string."""
    return (
        f"You are a person from {profile['country']}, "
        f"age group: {profile['age_group']}, "
        f"gender: {profile['gender']}, "
        f"education: {profile['education']}, "
        f"marital status: {profile['marital_status']}, "
        f"religion: {profile['religion']}, "
        f"ethnicity: {profile['ethnicity']}, "
        f"employment: {profile['employment']}."
    )

def format_demographics_facebook(row):
    """Format Facebook demographic information into a string."""
    parts = []
    if row.get('annotator_age'):
        parts.append(f"Age: {row['annotator_age']}")
    if row.get('annotator_gender'):
        parts.append(f"Gender: {row['annotator_gender']}")
    if row.get('annotator_country'):
        parts.append(f"Country: {row['annotator_country']}")
    if row.get('annotator_political'):
        parts.append(f"Political: {row['annotator_political']}")
    if row.get('annotator_ethnicity'):
        parts.append(f"Ethnicity: {row['annotator_ethnicity']}")
    if row.get('annotator_education_level'):
        parts.append(f"Education: {row['annotator_education_level']}")
    return ", ".join(parts)

def format_demographics_prism(row):
    """Format PRISM demographic information into a string."""
    parts = []
    # Country
    if row.get('location') and isinstance(row['location'], dict) and row['location'].get('reside_country'):
        parts.append(f"Country: {row['location']['reside_country']}")
    
    # Age
    if row.get('age'):
        parts.append(f"Age: {row['age']}")
        
    # Gender
    if row.get('gender'):
        parts.append(f"Gender: {row['gender']}")
        
    # Education
    if row.get('education'):
        parts.append(f"Education: {row['education']}")
        
    # Religion
    if row.get('religion') and isinstance(row['religion'], dict) and row['religion'].get('simplified'):
        val = row['religion']['simplified']
        if val and val.lower() not in ['prefer not to say', 'none', 'other']:
             parts.append(f"Religion: {val}")
             
    # Ethnicity
    if row.get('ethnicity') and isinstance(row['ethnicity'], dict) and row['ethnicity'].get('simplified'):
        val = row['ethnicity']['simplified']
        if val and val.lower() not in ['prefer not to say', 'other']:
            parts.append(f"Ethnicity: {val}")
            
    # Employment
    if row.get('employment_status'):
        parts.append(f"Employment: {row['employment_status']}")
        
    return ", ".join(parts)

def build_wvs_prompt(question_id: str, codebook: Dict):
    """Build user prompt for a specific WVS question."""
    q_info = codebook.get(question_id, {})
    question_text = q_info.get('question', '')
    choices = q_info.get('choices', {})
    
    if not question_text or not choices:
        return None
    
    valid_choices = {k: v for k, v in choices.items() if int(k) >= 0}
    if not valid_choices:
        return None
    
    sorted_numeric = sorted(valid_choices.keys(), key=lambda x: int(x))
    letter_map = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']
    choice_lines = []
    numeric_to_letter = {}
    
    for i, numeric_choice in enumerate(sorted_numeric):
        letter = letter_map[i]
        choice_desc = valid_choices[numeric_choice]
        choice_lines.append(f"{letter}. {choice_desc}")
        numeric_to_letter[numeric_choice] = letter

    choices_text = " ".join(choice_lines)
    prompt = f"{question_text}? {choices_text}. You can only choose one option."
    numeric_choices_list = [int(n) for n in sorted_numeric]
    
    return prompt, numeric_to_letter, numeric_choices_list

def is_valid_answer(answer_value) -> bool:
    if pd.isna(answer_value): return False
    if isinstance(answer_value, (int, float)): return int(answer_value) >= 0
    if isinstance(answer_value, str) and answer_value.strip().lstrip('-').isdigit():
        return int(answer_value) >= 0
    return False

# --- Main Processing Functions ---

def process_and_save_chunk(args):
    df_chunk, codebook, question_columns, model_name, output_file, mode = args
    
    # Initialize tokenizer inside the worker
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    system_prompt = "You are a helpful assistant that answers survey questions honestly."
    
    with gzip.open(output_file, 'wt', encoding='utf-8') as f:  # Use gzip compression
        for _, row in df_chunk.iterrows():
            # Extract demographics
            profile_dict = extract_demographic_profile(row)
            demographic_str = format_demographics_dict(profile_dict)
            
            participant_id = row.name # Assuming index is D_INTERVIEW
            
            for q_id in question_columns:
                if q_id not in row: 
                    continue
                    
                answer_value = row[q_id]
                
                if not is_valid_answer(answer_value):
                    continue
                    
                prompt_result = build_wvs_prompt(q_id, codebook)
                if prompt_result is None:
                    continue
                    
                user_prompt, numeric_to_letter, numeric_choices = prompt_result
                letter_answer = numeric_to_letter.get(str(int(answer_value)))
                if letter_answer is None:
                    continue
                
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": letter_answer}
                ]

                # Apply Chat Template based on mode
                if mode == 'test':
                    messages_for_template = messages[:-1] # Remove assistant message
                    text = tokenizer.apply_chat_template(
                        messages_for_template, 
                        tokenize=False, 
                        add_generation_prompt=True,
                        enable_thinking=False
                    )
                else:
                    text = tokenizer.apply_chat_template(
                        messages, 
                        tokenize=False, 
                        add_generation_prompt=False
                    )
                
                record = {
                    "text": text,
                    "source": user_prompt,
                    "target": letter_answer,
                    "demographic": demographic_str,
                    "numeric_choices": numeric_choices,
                    "letter_choices": list(numeric_to_letter.values()),
                    "question_id": q_id,
                    "participant_id": participant_id,
                    "messages": messages
                }
                
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    return output_file

def process_wvs_split(mode: str, output_dir: str):
    print(f"--- Processing WVS Dataset ({mode}) ---")
    codebook, demographic_df, value_df = load_wvs_raw_data(mode)
    
    # Identify question columns
    question_columns = [col for col in value_df.columns if col.startswith('Q')]
    
    print("Merging demographic and value dataframes...")
    demo_cols_needed = ['B_COUNTRY', 'X003R', 'Q260', 'Q275', 'Q273', 'Q289', 'Q290', 'Q279']
    demo_cols_exist = [c for c in demo_cols_needed if c in demographic_df.columns]
    
    # We want to keep rows from value_df that are in demographic_df.
    # value_df is the left one.
    full_df = value_df.join(demographic_df[demo_cols_exist], how='inner')
    # Note: value_df index is D_INTERVIEW. join preserves left index order for matches.
    
    # Free memory
    del value_df, demographic_df
    import gc
    gc.collect()
    
    print(f"Total participants to process: {len(full_df)}")
    
    # Prepare for multiprocessing
    num_proc = min(4, os.cpu_count() or 1) # Reduce to 4 processes to save memory and temp space
    chunk_size = int(np.ceil(len(full_df) / (num_proc * 2))) # Fewer chunks
    chunks = [full_df.iloc[i:i + chunk_size] for i in range(0, len(full_df), chunk_size)]
    
    # Use autodl-tmp for temp files (faster, compressed files should fit in remaining space)
    temp_dir = tempfile.mkdtemp(dir="/root/autodl-tmp")
    print(f"Using temporary directory: {temp_dir} (compressed files, should fit in ~13GB remaining)")
    
    tasks = []
    for i, chunk in enumerate(chunks):
        output_file = os.path.join(temp_dir, f"chunk_{i:05d}.jsonl.gz") # Use .gz extension
        tasks.append((chunk, codebook, question_columns, MODEL_NAME, output_file, mode))
    
    print(f"Starting processing with {num_proc} processes...")
    
    with multiprocessing.Pool(processes=num_proc) as pool:
        # Use imap_unordered to show progress
        results = list(tqdm(pool.imap_unordered(process_and_save_chunk, tasks), total=len(tasks)))
    
    print("Processing complete. Loading JSONL files...")
    
    # Load all JSONL files into a HuggingFace Dataset
    # Sort files by chunk index to ensure deterministic order
    json_files = sorted(
        [os.path.join(temp_dir, f) for f in os.listdir(temp_dir) if f.endswith('.jsonl.gz')]
    )
    
    if not json_files:
        print("Error: No output files generated!")
        return None
        
    ds = load_dataset("json", data_files=json_files, split="train")
    
    # Free memory by deleting large dataframes
    del full_df
    import gc
    gc.collect()
    
    print(f"Processed {len(ds)} WVS samples for {mode}.")

    # Add embeddings in batches to save memory
    print("Adding demographic embeddings in batches...")
    batch_size = 50000  # Process 50k samples at a time to save memory
    processed_datasets = []
    
    for i in range(0, len(ds), batch_size):
        batch_ds = ds.select(range(i, min(i + batch_size, len(ds))))
        batch_ds = add_demographic_embeddings(batch_ds)
        processed_datasets.append(batch_ds)
        print(f"Processed batch {i//batch_size + 1}/{(len(ds) + batch_size - 1)//batch_size}")
    
    # Concatenate all processed batches
    if len(processed_datasets) == 1:
        ds = processed_datasets[0]
    else:
        from datasets import concatenate_datasets
        ds = concatenate_datasets(processed_datasets)

    print("Saving as HuggingFace dataset...")
    ds.save_to_disk(output_dir)
    
    # Cleanup immediately to free space
    shutil.rmtree(temp_dir)
    print(f"WVS {mode} Done!")
    
    # Free memory
    del ds
    import gc
    gc.collect()

def sample_wvs_dataset(train_size=10000, test_size=1000):
    print("\n--- Sampling WVS Dataset ---")
    # Load full datasets
    train_path = os.path.join(WVS_FULL_OUTPUT_DIR, 'train')
    test_path = os.path.join(WVS_FULL_OUTPUT_DIR, 'test')
    
    print(f"Loading full train from {train_path}")
    full_train = load_from_disk(train_path)
    print(f"Loading full test from {test_path}")
    full_test = load_from_disk(test_path)
    
    # Sample 10k for train
    print(f"Sampling {train_size} for train...")
    sampled_train = full_train.shuffle(seed=42).select(range(min(train_size, len(full_train))))
    
    # Sample 1k for test
    print(f"Sampling {test_size} for test...")
    sampled_test = full_test.shuffle(seed=42).select(range(min(test_size, len(full_test))))
    
    # Save sampled datasets into a directory named by the sample size
    os.makedirs(WVS_SAMPLED_OUTPUT_DIR, exist_ok=True)
    sample_dir = os.path.join(WVS_SAMPLED_OUTPUT_DIR, f"sampled_{train_size//1000}k")
    os.makedirs(sample_dir, exist_ok=True)
    
    train_out = os.path.join(sample_dir, 'train')
    test_out = os.path.join(sample_dir, 'test')
    
    print(f"Saving sampled train to {train_out}")
    sampled_train.save_to_disk(train_out)
    
    print(f"Saving sampled test to {test_out}")
    sampled_test.save_to_disk(test_out)
    
    print("Sampling Done!")

def process_wvs():
    # Process both splits
    process_wvs_split('train', os.path.join(WVS_FULL_OUTPUT_DIR, 'train'))
    process_wvs_split('test', os.path.join(WVS_FULL_OUTPUT_DIR, 'test'))

def process_facebook():
    print("\n--- Processing Facebook Dataset (SFT & DPO) ---")
    print(f"Loading Facebook dataset from {FACEBOOK_INPUT_PATH}...")
    ds = load_from_disk(FACEBOOK_INPUT_PATH)
    
    # Initialize tokenizer
    print(f"Loading tokenizer from {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    sft_data = []
    dpo_data = []
    
    print("Processing Facebook conversations...")
    for row in tqdm(ds):
        demographic_str = format_demographics_facebook(row)
        
        # Initialize conversation history with system prompt
        # Note: We don't inject demographic here, it's handled by the dataset class if needed.
        # But for "Golden Thread", we need to maintain the history list.
        system_content = f"User Profile: {demographic_str}" if demographic_str else "You are a helpful assistant."
        history_messages = [{"role": "system", "content": system_content}]
        
        turns = ['first', 'second', 'third', 'fourth']
        
        for turn in turns:
            prompt_key = f"{turn}_turn_prompt"
            pref_key = f"{turn}_turn_preferred_response"
            
            prompt = row.get(prompt_key)
            pref_resp_suffix = row.get(pref_key)
            
            if not prompt or not pref_resp_suffix:
                break
                
            # Identify chosen response
            chosen_col_name = f"{turn}_turn_{pref_resp_suffix}"
            chosen_response = row.get(chosen_col_name)
            
            if not chosen_response:
                break
            
            # --- Build SFT Sample ---
            # SFT Sample: History + Current User Prompt + Chosen Response
            # We construct the full conversation up to this point
            sft_messages = history_messages + [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": chosen_response}
            ]
            
            # Apply chat template for SFT text
            text = tokenizer.apply_chat_template(
                sft_messages,
                tokenize=False,
                add_generation_prompt=False
            )
            
            sft_data.append({
                "text": text,
                "messages": sft_messages,
                "demographic": demographic_str,
                "conversation_id": row.get('conversation_id'),
                "turn": turn,
                "source": "facebook"
            })
            
            # --- Build DPO Samples ---
            # DPO Prompt: History + Current User Prompt
            
            dpo_prompt_messages = history_messages + [{"role": "user", "content": prompt}]
            
            # Apply chat template for DPO prompt
            dpo_prompt = tokenizer.apply_chat_template(
                dpo_prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
            
            candidate_suffixes = ['response_a', 'response_b', 'response_c', 'response_d']
            
            for suffix in candidate_suffixes:
                if suffix == pref_resp_suffix:
                    continue # Skip the chosen one
                
                rejected_col_name = f"{turn}_turn_{suffix}"
                rejected_response = row.get(rejected_col_name)
                
                if rejected_response:
                    dpo_data.append({
                        "prompt": dpo_prompt,
                        "chosen": chosen_response,
                        "rejected": rejected_response,
                        "demographic": demographic_str,
                        "conversation_id": row.get('conversation_id'),
                        "turn": turn,
                        "source": "facebook"
                    })
            
            # --- Update Golden Thread History ---
            # Crucial: Only append the CHOSEN response to history for the next turn
            history_messages.append({"role": "user", "content": prompt})
            history_messages.append({"role": "assistant", "content": chosen_response})

    print(f"Processed {len(sft_data)} Facebook SFT samples.")
    print(f"Processed {len(dpo_data)} Facebook DPO samples.")
    
    # Save SFT Data
    sft_output_dir = os.path.join(BASE_OUTPUT_DIR, "facebook_sft")
    print(f"Saving SFT data to {sft_output_dir}...")
    sft_ds = Dataset.from_list(sft_data)
    sft_ds = sft_ds.train_test_split(test_size=0.1, seed=42)

    # Add embeddings
    print("Adding demographic embeddings to SFT...")
    sft_ds = add_demographic_embeddings(sft_ds)

    sft_ds.save_to_disk(sft_output_dir)
    
    # Save DPO Data
    dpo_output_dir = os.path.join(BASE_OUTPUT_DIR, "facebook_dpo")
    print(f"Saving DPO data to {dpo_output_dir}...")
    dpo_ds = Dataset.from_list(dpo_data)
    dpo_ds = dpo_ds.train_test_split(test_size=0.1, seed=42)

    # Add embeddings
    print("Adding demographic embeddings to DPO...")
    dpo_ds = add_demographic_embeddings(dpo_ds)

    dpo_ds.save_to_disk(dpo_output_dir)
    
    print("Facebook Processing Done!")

def sample_facebook_dataset(sft_size=10000):
    """Sample a subset of the Facebook dataset (SFT & DPO)."""
    print(f"\n--- Sampling Facebook Dataset (SFT Size: {sft_size}) ---")
    
    sft_input_path = os.path.join(BASE_OUTPUT_DIR, "facebook_sft")
    dpo_input_path = os.path.join(BASE_OUTPUT_DIR, "facebook_dpo")
    
    sft_output_path = os.path.join(BASE_OUTPUT_DIR, f"facebook_sft_sampled_{sft_size}")
    dpo_output_path = os.path.join(BASE_OUTPUT_DIR, f"facebook_dpo_sampled_for_sft_{sft_size}")
    
    # Load SFT
    print(f"Loading SFT dataset from {sft_input_path}...")
    try:
        sft_ds = load_from_disk(sft_input_path)
    except FileNotFoundError:
        print("SFT dataset not found. Run process_facebook() first.")
        return

    if isinstance(sft_ds, dict):
        full_sft_train = sft_ds['train']
        full_sft_test = sft_ds['test']
    else:
        full_sft_train = sft_ds
        full_sft_test = None

    print(f"Original SFT Train Size: {len(full_sft_train)}")
    
    # Sample SFT
    # We want to keep the conversation IDs to filter DPO
    # Note: Facebook dataset is turn-based in this list, but we have conversation_id.
    # If we sample randomly, we might get partial conversations.
    # Ideally we sample by conversation_id.
    
    unique_conv_ids = list(set(full_sft_train['conversation_id']))
    import random
    random.seed(42)
    sampled_conv_ids = set(random.sample(unique_conv_ids, min(len(unique_conv_ids), sft_size)))
    
    print(f"Sampled {len(sampled_conv_ids)} unique conversations.")
    
    sampled_sft_train = full_sft_train.filter(lambda x: x['conversation_id'] in sampled_conv_ids)
    print(f"Sampled SFT Train Size (Turns): {len(sampled_sft_train)}")
    
    # Load DPO
    print(f"Loading DPO dataset from {dpo_input_path}...")
    try:
        dpo_ds = load_from_disk(dpo_input_path)
    except FileNotFoundError:
        print("DPO dataset not found. Run process_facebook() first.")
        return

    if isinstance(dpo_ds, dict):
        full_dpo_train = dpo_ds['train']
        full_dpo_test = dpo_ds['test']
    else:
        full_dpo_train = dpo_ds
        full_dpo_test = None
        
    print(f"Original DPO Train Size: {len(full_dpo_train)}")

    # Filter DPO based on sampled conversation IDs
    print("Filtering DPO dataset to match sampled conversations...")
    sampled_dpo_train = full_dpo_train.filter(lambda x: x['conversation_id'] in sampled_conv_ids)
    
    print(f"Filtered DPO Train Size: {len(sampled_dpo_train)}")
    
    # Save Datasets
    print(f"Saving sampled SFT to {sft_output_path}...")
    if isinstance(sft_ds, dict):
        from datasets import DatasetDict
        # Keep original test set for consistency
        sampled_sft_ds = DatasetDict({'train': sampled_sft_train, 'test': full_sft_test})
        sampled_sft_ds.save_to_disk(sft_output_path)
    else:
        sampled_sft_train.save_to_disk(sft_output_path)
        
    print(f"Saving sampled DPO to {dpo_output_path}...")
    if isinstance(dpo_ds, dict):
        from datasets import DatasetDict
        # Filter test set to match SFT test set conversations
        if full_sft_test is not None:
            sft_test_ids = set(full_sft_test['conversation_id'])
            sampled_dpo_test = full_dpo_test.filter(lambda x: x['conversation_id'] in sft_test_ids)
        else:
            sampled_dpo_test = full_dpo_test
            
        sampled_dpo_ds = DatasetDict({'train': sampled_dpo_train, 'test': sampled_dpo_test})
        sampled_dpo_ds.save_to_disk(dpo_output_path)
    else:
        sampled_dpo_train.save_to_disk(dpo_output_path)
        
    print("Sampling Done!")

def sample_facebook_eval_dataset(train_size=10000, test_size=1000):
    """Sample a subset of the Facebook benchmark dataset for evaluation."""
    print(f"\n--- Sampling Facebook Dataset (Train: {train_size}, Test: {test_size}) ---")
    input_path = FACEBOOK_INPUT_PATH
    output_dir = os.path.join(BASE_OUTPUT_DIR, "facebook_sampled")
    
    print(f"Loading Facebook benchmark dataset from {input_path}...")
    try:
        ds = load_from_disk(input_path)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Shuffle and split
    ds = ds.shuffle(seed=42)
    
    # Ensure we have enough samples
    total_needed = train_size + test_size
    if len(ds) < total_needed:
        print(f"Warning: Dataset only has {len(ds)} samples, which is less than {total_needed}.")
        train_size = int(len(ds) * 0.9)
        test_size = len(ds) - train_size
        print(f"Adjusted to Train: {train_size}, Test: {test_size}")

    sampled_train = ds.select(range(train_size))
    sampled_test = ds.select(range(train_size, train_size + test_size))
    
    print(f"Sampled Train Size: {len(sampled_train)}")
    print(f"Sampled Test Size: {len(sampled_test)}")
    
    os.makedirs(output_dir, exist_ok=True)
    sampled_train.save_to_disk(os.path.join(output_dir, "train"))
    sampled_test.save_to_disk(os.path.join(output_dir, "test"))
    
    print(f"Saved sampled eval dataset to {output_dir}")

def process_prism():
    print("\n--- Processing PRISM Dataset (SFT & DPO) ---")
    
    # Load Datasets
    print(f"Loading PRISM conversations from {PRISM_CONVERSATIONS_PATH}...")
    try:
        ds_conv = load_from_disk(PRISM_CONVERSATIONS_PATH)
    except:
        # Fallback if it's not a disk dataset but arrow files
        ds_conv = load_dataset("arrow", data_files=os.path.join(PRISM_CONVERSATIONS_PATH, "*.arrow"), split="train")

    print(f"Loading PRISM survey from {PRISM_SURVEY_PATH}...")
    try:
        ds_survey = load_from_disk(PRISM_SURVEY_PATH)
    except:
        ds_survey = load_dataset("arrow", data_files=os.path.join(PRISM_SURVEY_PATH, "*.arrow"), split="train")

    # Create User Profile Map
    print("Building user profile map...")
    user_profiles = {}
    for row in ds_survey:
        user_id = row['user_id']
        profile_str = format_demographics_prism(row)
        user_profiles[user_id] = profile_str
        
    # Initialize Tokenizer
    print(f"Loading tokenizer from {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    sft_data = []
    dpo_data = []
    
    print("Processing conversations...")
    for row in tqdm(ds_conv):
        user_id = row['user_id']
        profile_str = user_profiles.get(user_id, "")
        
        # Construct System Prompt
        system_content = f"User Profile: {profile_str}" if profile_str else "You are a helpful assistant."
        
        history = row['conversation_history']
        
        # Group by turn
        turns = {}
        for item in history:
            t = item['turn']
            if t not in turns:
                turns[t] = {'user': None, 'models': []}
            
            if item['role'] == 'user':
                turns[t]['user'] = item['content']
            elif item['role'] == 'model':
                turns[t]['models'].append(item)
        
        # Sort turns
        sorted_turn_ids = sorted(turns.keys())
        
        # Build Cumulative Context
        current_messages = [{"role": "system", "content": system_content}]
        
        valid_conversation = True
        
        for t in sorted_turn_ids:
            turn_data = turns[t]
            user_content = turn_data['user']
            
            if not user_content:
                # Skip if no user content for this turn (shouldn't happen usually)
                valid_conversation = False
                break
            
            # Add User Message to History
            current_messages.append({"role": "user", "content": user_content})
            
            # Find Chosen and Rejected
            chosen_item = None
            rejected_items = []
            
            for m in turn_data['models']:
                if m.get('if_chosen'):
                    chosen_item = m
                else:
                    rejected_items.append(m)
            
            if not chosen_item:
                # If no chosen item, we can't continue the chain for SFT or DPO history
                valid_conversation = False
                break
                
            # --- DPO Data Generation ---
            # Prompt is current_messages (System + History + Current User)
            dpo_prompt = tokenizer.apply_chat_template(
                current_messages, 
                tokenize=False, 
                add_generation_prompt=True,
                enable_thinking=False
            )
            
            for rejected in rejected_items:
                dpo_data.append({
                    "prompt": dpo_prompt,
                    "chosen": chosen_item['content'],
                    "rejected": rejected['content'],
                    "demographic": profile_str,
                    "conversation_id": row['conversation_id'],
                    "turn": t
                })
            
            # --- Update History for Next Turn / SFT ---
            current_messages.append({"role": "assistant", "content": chosen_item['content']})
        
        # --- SFT Data Generation ---
        # Only if we have a valid full conversation (or at least some turns)
        if len(current_messages) > 1: # At least System + User + Assistant
            # Apply chat template for the whole conversation
            text = tokenizer.apply_chat_template(
                current_messages,
                tokenize=False,
                add_generation_prompt=False
            )
            
            sft_data.append({
                "text": text,
                "demographic": profile_str,
                "conversation_id": row['conversation_id'],
                "messages": current_messages
            })

    # Save SFT Dataset
    print(f"Processed {len(sft_data)} PRISM SFT samples.")
    sft_ds = Dataset.from_list(sft_data)
    sft_ds = sft_ds.train_test_split(test_size=0.1, seed=42)

    # Add embeddings
    print("Adding demographic embeddings to SFT...")
    sft_ds = add_demographic_embeddings(sft_ds)

    print(f"Saving SFT dataset to {PRISM_SFT_OUTPUT_DIR}...")
    sft_ds.save_to_disk(PRISM_SFT_OUTPUT_DIR)
    
    # Save DPO Dataset
    print(f"Processed {len(dpo_data)} PRISM DPO samples.")
    dpo_ds = Dataset.from_list(dpo_data)
    dpo_ds = dpo_ds.train_test_split(test_size=0.1, seed=42)

    # Add embeddings
    print("Adding demographic embeddings to DPO...")
    dpo_ds = add_demographic_embeddings(dpo_ds)

    print(f"Saving DPO dataset to {PRISM_DPO_OUTPUT_DIR}...")
    dpo_ds.save_to_disk(PRISM_DPO_OUTPUT_DIR)
    
    print("PRISM Processing Done!")

def sample_prism_dataset(sft_size=2000):
    print(f"\n--- Sampling PRISM Dataset (SFT Size: {sft_size}) ---")
    
    # Paths
    sft_input_path = PRISM_SFT_OUTPUT_DIR
    dpo_input_path = PRISM_DPO_OUTPUT_DIR
    
    sft_output_path = os.path.join(BASE_OUTPUT_DIR, f"prism_sampled_sft_{sft_size}")
    dpo_output_path = os.path.join(BASE_OUTPUT_DIR, f"prism_sampled_dpo_for_sft_{sft_size}")
    
    # Load SFT
    print(f"Loading SFT dataset from {sft_input_path}...")
    try:
        sft_ds = load_from_disk(sft_input_path)
    except FileNotFoundError:
        print("SFT dataset not found. Run process_prism() first.")
        return

    # Combine splits if necessary to sample from all, or just sample from train
    # Usually we sample from train.
    if isinstance(sft_ds, dict):
        full_sft_train = sft_ds['train']
        full_sft_test = sft_ds['test']
    else:
        full_sft_train = sft_ds
        full_sft_test = None

    print(f"Original SFT Train Size: {len(full_sft_train)}")
    
    # Sample SFT
    # We want to keep the conversation IDs to filter DPO
    sampled_sft_train = full_sft_train.shuffle(seed=42).select(range(min(sft_size, len(full_sft_train))))
    
    # Get sampled conversation IDs
    sampled_conv_ids = set(sampled_sft_train['conversation_id'])
    print(f"Sampled {len(sampled_conv_ids)} unique conversations.")
    
    # Load DPO
    print(f"Loading DPO dataset from {dpo_input_path}...")
    try:
        dpo_ds = load_from_disk(dpo_input_path)
    except FileNotFoundError:
        print("DPO dataset not found. Run process_prism() first.")
        return

    if isinstance(dpo_ds, dict):
        full_dpo_train = dpo_ds['train']
        full_dpo_test = dpo_ds['test']
    else:
        full_dpo_train = dpo_ds
        full_dpo_test = None
        
    print(f"Original DPO Train Size: {len(full_dpo_train)}")

    # Filter DPO based on sampled conversation IDs
    print("Filtering DPO dataset to match sampled conversations...")
    sampled_dpo_train = full_dpo_train.filter(lambda x: x['conversation_id'] in sampled_conv_ids)
    
    print(f"Filtered DPO Train Size: {len(sampled_dpo_train)}")
    
    # Calculate Max Lengths
    print("Calculating token lengths for max_length determination...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    def get_len(text):
        return len(tokenizer(text)['input_ids'])
        
    # SFT Lengths
    sft_lengths = [get_len(x['text']) for x in tqdm(sampled_sft_train, desc="SFT Lengths")]
    sft_p95 = np.percentile(sft_lengths, 95)
    sft_max = np.max(sft_lengths)
    print(f"SFT Lengths - Max: {sft_max}, P95: {sft_p95}, Mean: {np.mean(sft_lengths)}")
    
    # DPO Lengths (Prompt + Chosen)
    # DPO usually concatenates prompt + chosen for length check
    dpo_lengths = [get_len(x['prompt'] + x['chosen']) for x in tqdm(sampled_dpo_train, desc="DPO Lengths")]
    dpo_p95 = np.percentile(dpo_lengths, 95)
    dpo_max = np.max(dpo_lengths)
    print(f"DPO Lengths (Prompt+Chosen) - Max: {dpo_max}, P95: {dpo_p95}, Mean: {np.mean(dpo_lengths)}")
    
    # Save Datasets
    print(f"Saving sampled SFT to {sft_output_path}...")
    # Re-wrap in DatasetDict if original was
    if isinstance(sft_ds, dict):
        from datasets import DatasetDict
        # For test set, we can either keep full test or sample it too. Let's keep full test for robust eval? 
        # Or sample proportional? Let's keep full test for now or small sample.
        # User didn't specify test size, let's just keep 10% of train size for test if we want consistency, 
        # or just keep the original test set. Let's keep original test set to be safe/comparable.
        sampled_sft_ds = DatasetDict({'train': sampled_sft_train, 'test': full_sft_test})
        sampled_sft_ds.save_to_disk(sft_output_path)
    else:
        sampled_sft_train.save_to_disk(sft_output_path)
        
    print(f"Saving sampled DPO to {dpo_output_path}...")
    if isinstance(dpo_ds, dict):
        from datasets import DatasetDict
        # Filter test set too?
        # If we want DPO test set to be consistent with SFT test set (which we kept full), we should keep full DPO test set?
        # Or filter DPO test set based on SFT test set conversations?
        # Let's filter DPO test set based on SFT test set conversations to be consistent.
        sft_test_ids = set(full_sft_test['conversation_id'])
        sampled_dpo_test = full_dpo_test.filter(lambda x: x['conversation_id'] in sft_test_ids)
        
        sampled_dpo_ds = DatasetDict({'train': sampled_dpo_train, 'test': sampled_dpo_test})
        sampled_dpo_ds.save_to_disk(dpo_output_path)
    else:
        sampled_dpo_train.save_to_disk(dpo_output_path)
        
    print("Sampling and Analysis Done!")


if __name__ == "__main__":
    # You can comment out one or the other to run separately
    process_wvs()
    # sample_wvs_dataset(10000, 1000)
    # sample_wvs_dataset(50000, 5000)
    sample_wvs_dataset(500000, 50000)
    process_facebook()
    # sample_facebook_dataset(10000)
    # sample_facebook_eval_dataset(10000, 1000)
    process_prism()
    # sample_prism_dataset(2000)
