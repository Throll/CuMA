#!/usr/bin/env bash
set -eu

export PYTHONPATH="."

# Default Configuration (Targeting PRISM DPO)
BASE_MODEL="../MyModels/Qwen3-8B"
EMBEDDING_MODEL="../MyModels/Qwen3-Embedding-0.6B"
SFT_MODEL="" # Optional: Path to SFT model to use as reference (e.g. ./checkpoints/sft_model)

ADAPTER_PATH="./checkpoints/cuma_prism_workflow_2k_dpo"
DATASET_PATH="./data/prism_sampled_dpo_for_sft_2000"
OUTPUT_FILE="./eval_results/cuma_prism_dpo_eval.json"
BASE_RESPONSES="./eval_results/base_prism_responses.json"

NUM_SAMPLES=50 # Set to -1 for all

# Judge Configuration
JUDGE_MODEL=""
API_KEY="sk-..."
BASE_URL=""

# GPU Setup
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
else
  export CUDA_VISIBLE_DEVICES=0
    echo "Setting CUDA_VISIBLE_DEVICES=0"
fi

# Check if adapter exists, warn if not (but proceed as script might be used for base model too if modified)
if [ ! -d "$ADAPTER_PATH" ]; then
    echo "WARNING: Adapter path $ADAPTER_PATH does not exist."
    echo "Please ensure training is complete or update ADAPTER_PATH."
fi

echo "Starting Evaluation..."
echo "Model: $BASE_MODEL"
echo "Adapter: $ADAPTER_PATH"
echo "Dataset: $DATASET_PATH"

python evaluate/evaluate_dpo.py \
    --model_name_or_path "${BASE_MODEL}" \
    --sft_model_path "${SFT_MODEL}" \
    --adapter_path "${ADAPTER_PATH}" \
    --embedding_model_name "${EMBEDDING_MODEL}" \
    --dataset_path "${DATASET_PATH}" \
    --eval_output_path "${OUTPUT_FILE}" \
    --base_responses_path "${BASE_RESPONSES}" \
    --num_samples ${NUM_SAMPLES} \
    --per_device_eval_batch_size 4 \
    --judge_model "${JUDGE_MODEL}" \
    --api_key "${API_KEY}" \
    --base_url "${BASE_URL}" \
    --output_dir "./eval_results/dpo_logs"
