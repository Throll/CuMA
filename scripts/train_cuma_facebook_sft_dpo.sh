#!/usr/bin/env bash
set -eu

# --- Configuration ---
# Paths
BASE_MODEL_PATH="../MyModels/Qwen3-8B"
EMBEDDING_MODEL_PATH="../MyModels/Qwen3-Embedding-0.6B"
SFT_DATASET="./data/facebook_sft_sampled_10000"
DPO_DATASET="./data/facebook_dpo_sampled_for_sft_10000"

# Experiment Names
EXP_PREFIX="cuma_facebook_workflow_10k"
SFT_EXP_NAME="${EXP_PREFIX}_sft"
DPO_EXP_NAME="${EXP_PREFIX}_dpo"

# Output Directories
CHECKPOINT_ROOT="./checkpoints"
SFT_OUTPUT_DIR="${CHECKPOINT_ROOT}/${SFT_EXP_NAME}"
DPO_OUTPUT_DIR="${CHECKPOINT_ROOT}/${DPO_EXP_NAME}"

# Training Hyperparameters
SFT_EPOCHS=1
DPO_EPOCHS=1
MAX_LENGTH=2048

# CuMa Hyperparameters
NUM_EXPERTS=8
TOP_K=2
DEMOGRAPHIC_EMBED_DIM=1024
LORA_R=8
LORA_ALPHA=16

# --- Environment Setup ---
export PYTHONPATH="."
export TOKENIZERS_PARALLELISM=false
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_IB_DISABLE=1

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
else
    if command -v nvidia-smi &> /dev/null; then
        NUM_GPUS=$(nvidia-smi -L | wc -l)
    else
        NUM_GPUS=1
    fi
fi

# DeepSpeed Config
ZERO_STAGE=1
DEEPSPEED_CONFIG="scripts/ds_config_zero${ZERO_STAGE}.json"

echo "=================================================="
echo "Starting Full CuMa Workflow (SFT -> DPO) for Community Alignment (Facebook)"
echo "Base Model: ${BASE_MODEL_PATH}"
echo "SFT Output: ${SFT_OUTPUT_DIR}"
echo "DPO Output: ${DPO_OUTPUT_DIR}"
echo "GPUs: ${NUM_GPUS}"
echo "=================================================="

# --- Step 1: SFT Training ---
echo ""
echo ">>> [Step 1/2] Starting SFT Training..."
echo "Dataset: ${SFT_DATASET}"

# SFT Batch Size Configuration
SFT_PER_DEVICE_BATCH_SIZE=2
SFT_TOTAL_BATCH_SIZE=16
SFT_GRAD_ACC=$((SFT_TOTAL_BATCH_SIZE / (SFT_PER_DEVICE_BATCH_SIZE * NUM_GPUS)))

if [ "$SFT_GRAD_ACC" -lt 1 ]; then
    SFT_GRAD_ACC=1
fi

accelerate launch \
    --num_processes $NUM_GPUS \
    --num_machines 1 \
    --mixed_precision bf16 \
    --dynamo_backend no \
    --use_deepspeed \
    --deepspeed_config_file "${DEEPSPEED_CONFIG}" \
    --zero_stage $ZERO_STAGE \
    train/train_cuma_sft.py \
    --model "${BASE_MODEL_PATH}" \
    --embedding_model "${EMBEDDING_MODEL_PATH}" \
    --dataset_path "${SFT_DATASET}" \
    --dataset_type "facebook" \
    --save_dir "${SFT_OUTPUT_DIR}" \
    --bf16 \
    --num_epochs ${SFT_EPOCHS} \
    --batch_size ${SFT_PER_DEVICE_BATCH_SIZE} \
    --accumulation_steps ${SFT_GRAD_ACC} \
    --lr 2e-5 \
    --weight_decay 0.01 \
    --logging_steps 10 \
    --save_steps 999999 \
    --lora_r ${LORA_R} \
    --lora_alpha ${LORA_ALPHA} \
    --num_experts ${NUM_EXPERTS} \
    --top_k ${TOP_K} \
    --demographic_embed_dim ${DEMOGRAPHIC_EMBED_DIM}

# --- Step 2: DPO Training (Initializing from SFT Adapter) ---
echo ""
echo ">>> [Step 2/2] Starting DPO Training..."
echo "Initializing from SFT Adapter: ${SFT_OUTPUT_DIR}"
echo "DPO Dataset: ${DPO_DATASET}"

# DPO Batch Size Configuration
DPO_PER_DEVICE_BATCH_SIZE=1
DPO_GRAD_ACC=1 # Adjust based on GPU memory

accelerate launch \
    --num_processes $NUM_GPUS \
    --num_machines 1 \
    --mixed_precision bf16 \
    --dynamo_backend no \
    --use_deepspeed \
    --deepspeed_config_file "${DEEPSPEED_CONFIG}" \
    --zero_stage $ZERO_STAGE \
    train/train_cuma_dpo.py \
    --model_name_or_path "${BASE_MODEL_PATH}" \
    --embedding_model_name_or_path "${EMBEDDING_MODEL_PATH}" \
    --adapter_path "${SFT_OUTPUT_DIR}" \
    --dataset_path "${DPO_DATASET}" \
    --output_dir "${DPO_OUTPUT_DIR}" \
    --overwrite_output_dir True \
    --per_device_train_batch_size ${DPO_PER_DEVICE_BATCH_SIZE} \
    --learning_rate 5e-6 \
    --beta 0.1 \
    --max_length ${MAX_LENGTH} \
    --max_prompt_length 1024 \
    --num_train_epochs ${DPO_EPOCHS} \
    --logging_steps 10 \
    --save_steps 999999 \
    --bf16 True \
    --remove_unused_columns False \
    --num_experts ${NUM_EXPERTS} \
    --top_k ${TOP_K} \
    --demographic_embed_dim ${DEMOGRAPHIC_EMBED_DIM} \
    --lora_r ${LORA_R} \
    --lora_alpha ${LORA_ALPHA} \
    --weight_decay 0.01

echo ""
echo ">>> Workflow Completed!"
echo "Final DPO Adapter saved at: ${DPO_OUTPUT_DIR}"
