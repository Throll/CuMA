# CuMA Implementation

This directory contains the source code for the CuMA framework.

## Requirements
The project uses `pyproject.toml` for dependency management. We recommend using [uv](https://github.com/astral-sh/uv) for fast installation:
```bash
uv sync
```
Key dependencies include `transformers`, `peft`, `deepspeed`, and `torch`.

## Main Components
- `models/cuma_model.py`: Implementation of the CuMA (Cultural Mixture of Adapters) layer and model.
- `train/`: Scripts for different training stages.
- `scripts/`: Shell scripts for orchestrating training and evaluation.
- `WorldValuesBench/`: Benchmark dataset placeholder. See `WorldValuesBench/README.md` for download instructions.

## Running Experiments

### 0. Dataset Preparation
- **WorldValuesBench (WVB)**: Download the repository from [https://github.com/Demon702/WorldValuesBench](https://github.com/Demon702/WorldValuesBench) and place its contents in the `WorldValuesBench/` directory.
- **PRISM & Community Alignment (CA)**: Run the provided download script to fetch these from Hugging Face:
  ```bash
  python download_datasets.py
  ```

### 1. Preprocessing
Run the scripts in `process_datasets/` to prepare the WVB, CA, and PRISM datasets.

### 2. Training
To train the CuMA model on WVS via SFT:
```bash
bash scripts/train_cuma_wvs_sft.sh
```

To run the full SFT+DPO workflow on Community Alignment (CA):
```bash
bash scripts/train_cuma_facebook_sft_dpo.sh
```

To run the full SFT+DPO workflow on PRISM:
```bash
bash scripts/train_cuma_prism_sft_dpo.sh
```

### 3. Evaluation
To evaluate the model on WorldValuesBench:
```bash
bash scripts/evaluate_wvs.sh
```

To evaluate DPO performance on PRISM or CA:
```bash
bash scripts/evaluate_dpo.sh
```
