import logging
from typing import List, Dict, Optional
from .base_dataset import BaseDataset


class WVSDataset(BaseDataset):
    """World Values Survey dataset """

    def __init__(
        self,
        dataset_path: str,
        tokenizer,
        model_name_or_path: Optional[str] = None,
        include_demographic: bool = True,
        max_length: Optional[int] = None,
        use_demographic_system_prompt: bool = False,
        is_eval: bool = False
    ):
        self.is_eval = is_eval
        super().__init__(
            dataset_path=dataset_path,
            tokenizer=tokenizer,
            model_name_or_path=model_name_or_path,
            include_demographic=include_demographic,
            max_length=max_length,
            use_demographic_system_prompt=use_demographic_system_prompt
        )
        """Validate that data has required fields"""
        if not self.data:
            return
        
        sample = self.data[0]
        required_fields = ['text', 'source', 'target', 'letter_choices', 'numeric_choices']
        
        missing_fields = [field for field in required_fields if field not in sample]
        if missing_fields:
            raise ValueError(
                f"Dataset missing required fields: {missing_fields}. "
                f"Available fields: {list(sample.keys())}"
            )
        
        if self.include_demographic and 'demographic' not in sample:
            logging.warning(
                "include_demographic=True but 'demographic' field not found in data. "
                "Demographic info will not be available."
            )
            self.include_demographic = False
            
        if self.use_demographic_system_prompt:
            if 'demographic' not in sample:
                raise ValueError("use_demographic_system_prompt=True but 'demographic' field missing")
            if 'messages' not in sample:
                raise ValueError("use_demographic_system_prompt=True but 'messages' field missing")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def get_collate_fn(self):
        """Return collate function for DataLoader with dynamic padding"""
        
        def collate_fn(batch: List[Dict]) -> Dict:            
            if self.use_demographic_system_prompt:
                texts = []
                for item in batch:
                    # Create a copy of messages to modify
                    messages = [dict(m) for m in item['messages']]
                    
                    # Replace system prompt with demographic info
                    if messages and messages[0]['role'] == 'system':
                        messages[0]['content'] = item['demographic']
                    else:
                        raise
                        # messages.insert(0, {'role': 'system', 'content': item['demographic']})
                    
                    if self.is_eval:
                        # For evaluation, truncate messages to before the last assistant
                        last_assistant_idx = -1
                        for i, msg in enumerate(messages):
                            if msg['role'] == 'assistant':
                                last_assistant_idx = i
                        if last_assistant_idx == -1:
                            truncated_messages = messages
                        else:
                            truncated_messages = messages[:last_assistant_idx]
                        text = self.tokenizer.apply_chat_template(
                            truncated_messages,
                            tokenize=False,
                            add_generation_prompt=True
                        )
                        if "Qwen3" in (self.model_name_or_path or ""):
                            text += "<think>\n\n</think>\n\n"
                        # print(f"Eval text: {text}")
                        # exit(0)
                    else:
                        # For training, use full text
                        text = self.tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=False
                        )
                    texts.append(text)
            else:
                if self.is_eval:
                    texts = []
                    for item in batch:
                        messages = item['messages']
                        # For evaluation, truncate messages to before the last assistant
                        last_assistant_idx = -1
                        for i, msg in enumerate(messages):
                            if msg['role'] == 'assistant':
                                last_assistant_idx = i
                        if last_assistant_idx == -1:
                            truncated_messages = messages
                        else:
                            truncated_messages = messages[:last_assistant_idx]
                        text = self.tokenizer.apply_chat_template(
                            truncated_messages,
                            tokenize=False,
                            add_generation_prompt=True
                        )
                        if "Qwen3" in (self.model_name_or_path or ""):
                            text += "<think>\n\n</think>\n\n"
                        texts.append(text)
                else:
                    texts = [item['text'] for item in batch]
            
            encodings = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt',
                add_special_tokens=False
            )
            
            labels = encodings['input_ids'].clone()
            
            # Robust masking: find assistant marker token subsequence inside encoded input_ids
            for i in range(len(texts)):
                input_ids_row = encodings['input_ids'][i].tolist()

                # Try primary marker, then fallbacks
                # Order matters: we want to match the most specific marker first if we want to mask more,
                # but here we want to match the start of the assistant response.
                candidate_markers = [
                    self.assistant_marker,
                    '<|im_start|>assistant\n',
                    '<|start_header_id|>assistant<|end_header_id|>\n\n',
                ]

                found = False
                for marker in candidate_markers:
                    if not marker: continue
                    try:
                        marker_tokens = self.tokenizer.encode(marker, add_special_tokens=False)
                    except Exception:
                        marker_tokens = []
                    if not marker_tokens:
                        continue

                    # search for last occurrence of marker_tokens in input_ids_row
                    mlen = len(marker_tokens)
                    for start in range(len(input_ids_row) - mlen, -1, -1):
                        if input_ids_row[start:start+mlen] == marker_tokens:
                            prefix_len = start + mlen
                            labels[i, :prefix_len] = -100
                            found = True
                            break
                    if found:
                        break

                if not found:
                    raise ValueError(f"Assistant marker tokens not found in encoded input_ids for sample {i}")
            
            # Mask padding tokens
            labels[labels == self.tokenizer.pad_token_id] = -100
            
            trainable_tokens = (labels != -100).sum()
            # if trainable_tokens == 0:
            #     raise ValueError("No trainable tokens found in batch. Check data formatting.")
            
            result = {
                'input_ids': encodings['input_ids'],
                'attention_mask': encodings['attention_mask'],
                'labels': labels,
                'target': [item['target'] for item in batch],
                'letter_choices': [item.get('letter_choices') for item in batch],
                'numeric_choices': [item.get('numeric_choices') for item in batch],
                'question_id': [item.get('question_id', -1) for item in batch],
                'participant_id': [item.get('participant_id', -1) for item in batch]
            }
            
            if self.include_demographic:
                result['demographic'] = [item['demographic'] for item in batch]
                
                # Check for pre-computed embeddings
                if 'demographic_embed' in batch[0]:
                    import torch
                    import numpy as np
                    
                    embeds = []
                    for item in batch:
                        e = item['demographic_embed']
                        if isinstance(e, list):
                            e = torch.tensor(e, dtype=torch.float32)
                        elif isinstance(e, np.ndarray):
                            e = torch.from_numpy(e).float()
                        elif isinstance(e, torch.Tensor):
                            e = e.float()
                        embeds.append(e)
                    
                    result['demographic_embed'] = torch.stack(embeds)
            
            return result
        
        return collate_fn
