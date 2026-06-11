#!/usr/bin/env bash
set -eu

# source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="."
export TOKENIZERS_PARALLELISM=false
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_IB_DISABLE=1


MODEL_PATH="../MyModels/Qwen3-0.6B"
DATASET_PATH="./data/wvs_sampled_sft/sampled_50k/train"

EXP_NAME="hydralora_wvs_50k"
SAVE_DIR="./checkpoints/${EXP_NAME}"

EVAL_ONLY=${EVAL_ONLY:-false}

TARGET_MODULES="q_proj,v_proj"

# Hyperparameters matching CuMA and MixLoRA
LORA_R=8
LORA_ALPHA=16
NUM_EXPERTS=8
TOP_K=2
LAMBDA_AUX=0.01
ROUTER_HIDDEN_DIM=256
NUM_ROUTER_LAYERS=2

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
else
    if command -v nvidia-smi &> /dev/null; then
        NUM_GPUS=$(nvidia-smi -L | wc -l)
    else
        NUM_GPUS=1
    fi
fi

TOTAL_BATCH_SIZE=16
PER_DEVICE_BATCH_SIZE=8
GRAD_ACC=$((TOTAL_BATCH_SIZE / (PER_DEVICE_BATCH_SIZE * NUM_GPUS)))

if [ "$EVAL_ONLY" = false ]; then
    echo "Starting HydraLoRA training..."
    accelerate launch \
        --num_processes $NUM_GPUS \
        --num_machines 1 \
        --mixed_precision bf16 \
        train/train_hydralora_sft.py \
        --model "$MODEL_PATH" \
        --data_path "$DATASET_PATH" \
        --save_dir "$SAVE_DIR" \
        --lora_r $LORA_R \
        --lora_alpha $LORA_ALPHA \
        --num_experts $NUM_EXPERTS \
        --top_k $TOP_K \
        --target_modules "$TARGET_MODULES" \
        --router_hidden_dim $ROUTER_HIDDEN_DIM \
        --num_router_mlp_layers $NUM_ROUTER_LAYERS \
        --lambda_auxiliary $LAMBDA_AUX \
        --batch_size $PER_DEVICE_BATCH_SIZE \
        --accumulation_steps $GRAD_ACC \
        --lr 2e-5 \
        --num_epochs 1 \
        --warmup_steps 100 \
        --save_steps 500 \
        --logging_steps 10 \
        --max_length 1024
fi

DATASET="./data/wvs_sampled_sft/sampled_50k/test"
PEFT_MODEL_PATH=${SAVE_DIR}

echo "Starting evaluation..."
accelerate launch \
    --num_processes $NUM_GPUS \
    --num_machines 1 \
    --mixed_precision bf16 \
    evaluate/evaluate_wvs.py \
    --model "$MODEL_PATH" \
    --dataset "$DATASET" \
    --eval_batch_size 32 \
    --bf16 \
    --seed 42 \
    --output_dir "./eval_results/$EXP_NAME" \
    --peft_model_path "$PEFT_MODEL_PATH" \
    --adapter_type hydralora
