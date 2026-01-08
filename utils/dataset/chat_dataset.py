from typing import List, Dict, Optional
import torch
from .base_dataset import BaseDataset

class ChatDataset(BaseDataset):
    """Dataset for chat-formatted data (list of messages)"""

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

    def _validate_data_format(self):
        if not self.data:
            return
        sample = self.data[0]
        if 'messages' not in sample:
            raise ValueError(f"Dataset missing 'messages' field. Available: {list(sample.keys())}")
        
        if self.include_demographic and 'demographic' not in sample:
            # For CuMa training, demographic info is crucial.
            # We can either warn or raise error. Given user requirement "demographic field is necessary", we raise error.
            raise ValueError(
                "include_demographic=True but 'demographic' field not found in data. "
                "This field is required for CuMa training."
            )

    def get_collate_fn(self):
        def collate_fn(batch: List[Dict]) -> Dict:
            texts = []
            for item in batch:
                messages = [dict(m) for m in item['messages']]
                
                if self.use_demographic_system_prompt and 'demographic' in item:
                    demographic_text = str(item['demographic'])
                    if messages and messages[0]['role'] == 'system':
                        messages[0]['content'] = demographic_text
                    else:
                        messages.insert(0, {'role': 'system', 'content': demographic_text})
                
                if self.is_eval:
                    # For evaluation, remove the last assistant message if it exists
                    if messages and messages[-1]['role'] == 'assistant':
                        messages = messages[:-1]
                    
                    text = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                else:
                    text = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=False
                    )
                texts.append(text)

            encodings = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt',
                add_special_tokens=False
            )
            
            labels = encodings['input_ids'].clone()
            
            # Mask padding
            labels[labels == self.tokenizer.pad_token_id] = -100
            
            # Mask user prompts (heuristic: mask until last assistant marker)
            # Robust masking: find assistant marker token subsequence inside encoded input_ids
            for i in range(len(texts)):
                input_ids_row = encodings['input_ids'][i].tolist()

                # Try primary marker, then fallbacks
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
            
            result = {
                'input_ids': encodings['input_ids'],
                'attention_mask': encodings['attention_mask'],
                'labels': labels
            }
            
            if self.include_demographic:
                result['demographic'] = [item.get('demographic') for item in batch]
                
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
