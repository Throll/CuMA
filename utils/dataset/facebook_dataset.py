from .chat_dataset import ChatDataset
import random
from torch.utils.data import Dataset
from datasets import load_from_disk

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

class FacebookDataset(ChatDataset):
    """
    Dataset for Facebook Community Alignment data.
    Inherits from ChatDataset as the data is processed into standard chat format.
    """
    pass

class FacebookDiscriminationDataset(Dataset):
    def __init__(self, dataset_path, tokenizer, mode="persona", few_shot_k=0, seed=42, max_samples=None):
        self.ds = load_from_disk(dataset_path)
        if max_samples is not None and max_samples < len(self.ds):
            # Use a fixed subset for evaluation
            indices = list(range(len(self.ds)))
            random.seed(seed)
            random.shuffle(indices)
            self.eval_indices = indices[:max_samples]
        else:
            self.eval_indices = list(range(len(self.ds)))
            
        self.tokenizer = tokenizer
        self.mode = mode # "vanilla", "persona"
        self.few_shot_k = few_shot_k
        self.seed = seed
        
        # For few-shot, we need a pool. We'll use the same dataset but exclude the current index.
        self.few_shot_pool_indices = list(range(len(self.ds)))

    def __len__(self):
        return len(self.eval_indices)

    def __getitem__(self, idx):
        real_idx = self.eval_indices[idx]
        row = self.ds[real_idx]
        demo_str = format_demographics_facebook(row)
        prompt = row['first_turn_prompt']
        
        choices = {
            'A': row['first_turn_response_a'],
            'B': row['first_turn_response_b'],
            'C': row['first_turn_response_c'],
            'D': row['first_turn_response_d']
        }
        
        pref = row['first_turn_preferred_response'] # e.g. "response_b"
        target_letter = pref.split('_')[-1].upper() # "B"
        
        # Construct messages
        messages = []
        
        # 1. System Prompt / Persona
        system_content = ""
        if self.mode == "persona":
            system_content = f"User Profile: {demo_str}\n"
        system_content += "You are a helpful assistant. Based on the user profile, choose the most appropriate response from the options provided."
        
        messages.append({"role": "system", "content": system_content})
        
        # 2. Few-shot examples
        if self.few_shot_k > 0:
            random.seed(self.seed + real_idx) # Ensure reproducible but different few-shot for each sample
            pool = [i for i in self.few_shot_pool_indices if i != real_idx]
            ex_indices = random.sample(pool, self.few_shot_k)
            for ex_idx in ex_indices:
                ex_row = self.ds[ex_idx]
                ex_demo = format_demographics_facebook(ex_row)
                ex_prompt = ex_row['first_turn_prompt']
                ex_choices = [
                    f"A: {ex_row['first_turn_response_a']}",
                    f"B: {ex_row['first_turn_response_b']}",
                    f"C: {ex_row['first_turn_response_c']}",
                    f"D: {ex_row['first_turn_response_d']}"
                ]
                ex_pref = ex_row['first_turn_preferred_response'].split('_')[-1].upper()
                
                ex_content = f"User Profile: {ex_demo}\nQuestion: {ex_prompt}\n\nOptions:\n" + "\n".join(ex_choices) + "\n\nAnswer:"
                messages.append({"role": "user", "content": ex_content})
                messages.append({"role": "assistant", "content": ex_pref})
        
        # 3. Current Question
        curr_choices = [
            f"A: {choices['A']}",
            f"B: {choices['B']}",
            f"C: {choices['C']}",
            f"D: {choices['D']}"
        ]
        curr_content = f"Question: {prompt}\n\nOptions:\n" + "\n".join(curr_choices) + "\n\nAnswer:"
        if self.mode == "vanilla":
            messages[0]['content'] = "You are a helpful assistant. Choose the most appropriate response from the options provided."

        messages.append({"role": "user", "content": curr_content})
        
        # Apply template
        full_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        if "Qwen3" in self.tokenizer.name_or_path:
             full_prompt += "<think>\n\n</think>\n\n"

        return {
            "full_prompt": full_prompt,
            "target": target_letter,
            "demographic": demo_str,
            "choices": ['A', 'B', 'C', 'D']
        }

    def get_collate_fn(self):
        def collate_fn(batch):
            prompts = [item['full_prompt'] for item in batch]
            targets = [item['target'] for item in batch]
            demos = [item['demographic'] for item in batch]
            
            encodings = self.tokenizer(
                prompts,
                padding=True,
                truncation=True,
                max_length=2048, # Facebook prompts can be long
                return_tensors='pt'
            )
            
            return {
                "input_ids": encodings['input_ids'],
                "attention_mask": encodings['attention_mask'],
                "targets": targets,
                "demographics": demos
            }
        return collate_fn
