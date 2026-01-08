#!/usr/bin/env bash
set -eu

export PYTHONPATH="."
export CUDA_VISIBLE_DEVICES=0

# Base Model
MODEL_PATH="../MyModels/Qwen3-8B"

# LoRA Adapter (Output from train_lora_prism_dpo.sh)
ADAPTER_PATH="./checkpoints/qwen3_0.6b_lora_prism_dpo_2k"

# Dataset (Test split will be used)
DATASET_PATH="./data/prism_sampled_dpo_for_sft_2000"

# Output
OUTPUT_FILE="./eval_results/prism_dpo_evaluation.json"

# OpenAI Config (Set these if you want to run the Judge)
# export OPENAI_API_KEY="sk-..."
# export OPENAI_BASE_URL="https://api.openai.com/v1"

echo "Evaluating PRISM DPO Model..."
echo "Base Model: $MODEL_PATH"
echo "Adapter: $ADAPTER_PATH"
echo "Dataset: $DATASET_PATH"

python evaluate/evaluate_dpo.py \
    --model_name_or_path "$MODEL_PATH" \
    --adapter_path "$ADAPTER_PATH" \
    --dataset_path "$DATASET_PATH" \
    --eval_output_path "$OUTPUT_FILE" \
    --num_samples 100 \
    --per_device_eval_batch_size 4 \
    --judge_model "gpt-4o"
