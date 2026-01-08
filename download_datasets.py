from datasets import load_dataset
import os

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
TARGET_DIR = os.path.join(CUR_DIR, "../MyDatasets")

def save_split(name, subset=None, split="train"):
    """Download and save dataset split with subset embedded in directory name"""
    ds = load_dataset(name, subset) if subset else load_dataset(name)
    
    # Build directory name: name/subset/split (Original Format)
    dir_name = name
    if subset:
        dir_name = os.path.join(dir_name, subset)
    if split:
        dir_name = os.path.join(dir_name, split)
    
    full_path = os.path.join(TARGET_DIR, dir_name)
    
    # save_to_disk automatically creates necessary subdirectories
    ds[split].save_to_disk(full_path)
    print(f"Saved {name} ({subset or 'default'}) {split} split to {full_path}")

# Download All Required Datasets
if __name__ == "__main__":
    print("Starting download of PRISM and Community Alignment datasets...")
    
    # PRISM Alignment Dataset
    save_split("HannahRoseKirk/prism-alignment", "conversations", "train")
    save_split("HannahRoseKirk/prism-alignment", "survey", "train")
    
    # Community Alignment Dataset (Facebook)
    save_split("facebook/community-alignment-dataset", "benchmark", "filtered")
    
    print("\nDownload complete. Datasets are saved in '../MyDatasets'.")
    print("You can now run 'python process_datasets/process_datasets.py' to prepare them.")
