import os
import json
import torch
import random
import numpy as np
from transformers import set_seed as transformers_seed
from typing import Optional


def set_seed(seed: Optional[int]):
    if seed is None:
        return
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    transformers_seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def to_json(data, path, file_name):
    if not os.path.exists(path):
        os.makedirs(path)
    with open(path + file_name, 'w') as f:
        json.dump(data, f)

def get_unique_dir(dir_path: str) -> str:
    # from datetime import datetime
    
    # if os.path.exists(dir_path):
    #     contents = os.listdir(dir_path)
    #     # If the directory is not empty and contains more than just a 'logs' folder
    #     if contents and not (len(contents) == 1 and contents[0] == 'logs'):
    #         timestamp = datetime.now().strftime("%Y%m%d%H%M")
    #         dir_path = f"{dir_path.rstrip(os.sep)}_{timestamp}"
    
    return dir_path


def clear_directory(dir_path: str):
    """Delete all contents of a directory if it exists."""
    import shutil
    if os.path.exists(dir_path):
        for item in os.listdir(dir_path):
            item_path = os.path.join(dir_path, item)
            try:
                if os.path.isfile(item_path) or os.path.islink(item_path):
                    os.unlink(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
            except Exception as e:
                print(f"Warning: could not delete {item_path}: {e}")
    else:
        os.makedirs(dir_path, exist_ok=True)
