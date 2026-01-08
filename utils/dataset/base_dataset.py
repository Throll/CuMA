import os
import gzip
import json
import logging
from typing import List, Dict, Optional
from datasets import load_from_disk
from torch.utils.data import Dataset

class BaseDataset(Dataset):
    """Base dataset class with common loading logic"""

    def __init__(
        self,
        dataset_path: str,
        tokenizer,
        model_name_or_path: Optional[str] = None,
        include_demographic: bool = True,
        max_length: Optional[int] = None,
        use_demographic_system_prompt: bool = False
    ):
        self.tokenizer = tokenizer
        self.model_name_or_path = model_name_or_path
        self.include_demographic = include_demographic
        self.max_length = max_length or tokenizer.model_max_length
        self.use_demographic_system_prompt = use_demographic_system_prompt
        self.data = []

        # Determine assistant marker based on model type
        if model_name_or_path and "Qwen3" in model_name_or_path:
            self.assistant_marker = '<|im_start|>assistant\n<think>\n\n</think>\n\n'
        elif model_name_or_path and "Llama-3.1" in model_name_or_path:
            self.assistant_marker = '<|start_header_id|>assistant<|end_header_id|>\n\n'
        else:
            self.assistant_marker = '<|im_start|>assistant\n'
        
        # Load data based on path type
        if os.path.isdir(dataset_path):
            self._load_huggingface_dataset(dataset_path)
        elif dataset_path.endswith('.parquet'):
            self._load_parquet(dataset_path)
        elif dataset_path.endswith('.gz'):
            self._load_gzipped_jsonl(dataset_path)
        elif dataset_path.endswith('.jsonl'):
            self._load_jsonl(dataset_path)
        else:
            raise ValueError(
                f"Unsupported dataset format: {dataset_path}. "
                "Expected directory (HF format), .parquet, .jsonl, or .jsonl.gz"
            )
        
        if not self.data:
            raise ValueError(f"No data loaded from {dataset_path}")
        
        self._validate_data_format()
        logging.info(f"Loaded {len(self.data)} samples from {dataset_path}")

    def _load_huggingface_dataset(self, path: str):
        """Load preprocessed HuggingFace Arrow format dataset"""
        logging.info(f"Loading HuggingFace dataset from {path}")
        hf_dataset = load_from_disk(path)
        
        if hasattr(hf_dataset, 'keys') and 'train' in hf_dataset:
            logging.info("Detected DatasetDict, using 'train' split")
            self.data = list(hf_dataset['train'])
        else:
            self.data = list(hf_dataset)

    def _load_parquet(self, path: str):
        """Load Parquet dataset"""
        logging.info(f"Loading Parquet dataset from {path}")
        from datasets import load_dataset
        # load_dataset with "parquet" builder
        hf_dataset = load_dataset("parquet", data_files=path, split="train")
        self.data = list(hf_dataset)
    
    def _load_gzipped_jsonl(self, path: str):
        """Load gzipped JSONL file"""
        logging.info(f"Loading gzipped JSONL from {path}")
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))
    
    def _load_jsonl(self, path: str):
        """Load plain JSONL file"""
        logging.info(f"Loading JSONL from {path}")
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))
    
    def _validate_data_format(self):
        """Override this in subclasses"""
        pass

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def get_collate_fn(self):
        raise NotImplementedError("Subclasses must implement get_collate_fn")
