#!/usr/bin/env bash
set -eu

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="."
export TOKENIZERS_PARALLELISM=false
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_IB_DISABLE=1

# source .venv/bin/activate

MODEL_PATH="../MyModels/Qwen3-8B"
EMBEDDING_MODEL_PATH="../MyModels/Qwen3-Embedding-0.6B"
DATASET_PATH="./data/facebook_dpo_sampled_10k"

EXP_NAME="cuma_facebook_dpo_10k"
OUTPUT_DIR="./checkpoints/${EXP_NAME}"

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
else
    if command -v nvidia-smi &> /dev/null; then
        NUM_GPUS=$(nvidia-smi -L | wc -l)
    else
        NUM_GPUS=1
    fi
fi

# Force 1 GPU for debugging
NUM_GPUS=1
export CUDA_VISIBLE_DEVICES=0

ZERO_STAGE=1
echo "Launching on $NUM_GPUS GPUs with DeepSpeed ZeRO-$ZERO_STAGE"
DEEPSPEED_CONFIG="scripts/ds_config_zero${ZERO_STAGE}.json"

# Ensure DeepSpeed config exists (create a temporary one if needed or rely on existing)
# Assuming scripts/ds_config_zero2.json exists as per workspace context implied by other scripts

accelerate launch \
    --num_processes $NUM_GPUS \
    --num_machines 1 \
    --mixed_precision bf16 \
    --dynamo_backend no \
    --use_deepspeed \
    --deepspeed_config_file "${DEEPSPEED_CONFIG}" \
    --zero_stage $ZERO_STAGE \
    train/train_cuma_dpo.py \
    --model_name_or_path "${MODEL_PATH}" \
    --embedding_model_name_or_path "${EMBEDDING_MODEL_PATH}" \
    --dataset_path "${DATASET_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --overwrite_output_dir True \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 5e-6 \
    --beta 0.1 \
    --max_length 2048 \
    --max_prompt_length 1024 \
    --num_train_epochs 0.0005 \
    --logging_steps 10 \
    --save_steps 999999 \
    --eval_steps 5000 \
    --eval_strategy steps \
    --bf16 True \
    --remove_unused_columns False \
    --report_to tensorboard \
    --run_name "${EXP_NAME}" \
    --save_safetensors False \
    --num_experts 8 \
    --top_k 2 \
    --demographic_embed_dim 1024 \
    --lora_r 8 \
    --lora_alpha 16 \
    --weight_decay 0.01