#!/usr/bin/env bash
set -eu

# --- Configuration ---
# Paths
BASE_MODEL_PATH="../MyModels/Qwen3-8B"
SFT_DATASET="./data/prism_sampled_sft_2000"
DPO_DATASET="./data/prism_sampled_dpo_for_sft_2000"

# Experiment Names
EXP_PREFIX="prism_workflow_2k"
SFT_EXP_NAME="lora_${EXP_PREFIX}_sft"
DPO_EXP_NAME="lora_${EXP_PREFIX}_dpo"

# Output Directories
CHECKPOINT_ROOT="./checkpoints"
SFT_OUTPUT_DIR="${CHECKPOINT_ROOT}/${SFT_EXP_NAME}"
DPO_OUTPUT_DIR="${CHECKPOINT_ROOT}/${DPO_EXP_NAME}"

# Training Hyperparameters
NUM_GPUS=1 # Auto-detected below
SFT_EPOCHS=1
DPO_EPOCHS=1
MAX_LENGTH=2048
SKIP_SFT=false # Set to true to skip SFT stage

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
echo "Starting Full LoRA Workflow (SFT -> DPO)"
echo "Base Model: ${BASE_MODEL_PATH}"
echo "SFT Output: ${SFT_OUTPUT_DIR}"
echo "DPO Output: ${DPO_OUTPUT_DIR}"
echo "GPUs: ${NUM_GPUS}"
echo "=================================================="

# --- Step 1: SFT Training ---
if [ "${SKIP_SFT:-false}" = "true" ]; then
    echo ">>> Skipping SFT Training (SKIP_SFT=true)..."
else
    echo ""
    echo ">>> [Step 1/2] Starting SFT Training..."
    echo "Dataset: ${SFT_DATASET}"

    # SFT Batch Size Configuration
    SFT_PER_DEVICE_BATCH_SIZE=4
    SFT_TOTAL_BATCH_SIZE=16
    SFT_GRAD_ACC=$((SFT_TOTAL_BATCH_SIZE / (SFT_PER_DEVICE_BATCH_SIZE * NUM_GPUS)))

    # Ensure accumulation steps is at least 1
    if [ "$SFT_GRAD_ACC" -lt 1 ]; then
      SFT_GRAD_ACC=1
    fi

    echo "SFT Configuration:"
    echo "  Per-Device Batch Size: ${SFT_PER_DEVICE_BATCH_SIZE}"
    echo "  Total Target Batch Size: ${SFT_TOTAL_BATCH_SIZE}"
    echo "  Gradient Accumulation Steps: ${SFT_GRAD_ACC}"

    accelerate launch \
        --num_processes $NUM_GPUS \
        --num_machines 1 \
        --mixed_precision bf16 \
        --dynamo_backend no \
        --use_deepspeed \
        --deepspeed_config_file "${DEEPSPEED_CONFIG}" \
        --zero_stage $ZERO_STAGE \
        train/train_lora_sft.py \
        --model "${BASE_MODEL_PATH}" \
        --dataset_path "${SFT_DATASET}" \
        --dataset_type "prism" \
        --save_dir "${SFT_OUTPUT_DIR}" \
        --bf16 \
        --num_epochs ${SFT_EPOCHS} \
        --batch_size ${SFT_PER_DEVICE_BATCH_SIZE} \
        --accumulation_steps ${SFT_GRAD_ACC} \
        --lr 2e-5 \
        --weight_decay 0.01 \
        --max_grad_norm 1.0 \
        --warmup_steps 0 \
        --schedule_name constant \
        --logging_steps 10 \
        --save_steps 999999 \
        --lora_r 8 \
        --lora_alpha 16 \
        --target_modules "q_proj v_proj" \
        --max_length ${MAX_LENGTH}

    echo ">>> SFT Training Complete!"
    echo "SFT Adapter saved to: ${SFT_OUTPUT_DIR}"
fi

# --- Step 2: DPO Training (Load and Continue) ---
echo ""
echo ">>> [Step 2/2] Starting DPO Training..."
echo "Loading Adapter from SFT: ${SFT_OUTPUT_DIR}"
echo "Dataset: ${DPO_DATASET}"

# DPO Batch Size Configuration
# DPO usually needs smaller batch size per device due to 2x forward passes (chosen/rejected)
DPO_PER_DEVICE_BATCH_SIZE=2
DPO_TOTAL_BATCH_SIZE=32
DPO_GRAD_ACC=$((DPO_TOTAL_BATCH_SIZE / (DPO_PER_DEVICE_BATCH_SIZE * NUM_GPUS)))

# Ensure accumulation steps is at least 1
if [ "$DPO_GRAD_ACC" -lt 1 ]; then
  DPO_GRAD_ACC=1
fi

echo "DPO Configuration:"
echo "  Per-Device Batch Size: ${DPO_PER_DEVICE_BATCH_SIZE}"
echo "  Total Target Batch Size: ${DPO_TOTAL_BATCH_SIZE}"
echo "  Gradient Accumulation Steps: ${DPO_GRAD_ACC}"

# Note: We use ZERO_STAGE=1 for DPO usually to be safe, or 2 if memory allows. 
# Keeping 2 as defined above or switching to 1 if unstable. Let's stick to the config used in separate script (Stage 1).
DPO_ZERO_STAGE=1
DPO_DS_CONFIG="scripts/ds_config_zero${DPO_ZERO_STAGE}.json"

accelerate launch \
    --num_processes $NUM_GPUS \
    --num_machines 1 \
    --mixed_precision bf16 \
    --dynamo_backend no \
    --use_deepspeed \
    --deepspeed_config_file "${DPO_DS_CONFIG}" \
    --zero_stage $DPO_ZERO_STAGE \
    train/train_lora_dpo.py \
    --model_name_or_path "${BASE_MODEL_PATH}" \
    --adapter_path "${SFT_OUTPUT_DIR}" \
    --dataset_path "${DPO_DATASET}" \
    --output_dir "${DPO_OUTPUT_DIR}" \
    --overwrite_output_dir True \
    --per_device_train_batch_size $DPO_PER_DEVICE_BATCH_SIZE \
    --per_device_eval_batch_size $DPO_PER_DEVICE_BATCH_SIZE \
    --gradient_accumulation_steps $DPO_GRAD_ACC \
    --learning_rate 5e-6 \
    --beta 0.1 \
    --max_length ${MAX_LENGTH} \
    --max_prompt_length 1024 \
    --num_train_epochs ${DPO_EPOCHS} \
    --logging_steps 100 \
    --save_steps 999999

echo ""
echo "=================================================="
echo "Workflow Complete!"
echo "Final DPO Model (Adapter) saved to: ${DPO_OUTPUT_DIR}"
echo "=================================================="
