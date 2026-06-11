#!/usr/bin/env bash
set -eu

source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="."
export TOKENIZERS_PARALLELISM=false

MODEL_PATH="../MyModels/Qwen3-8B"
EMBEDDING_MODEL_PATH="../MyModels/Qwen3-Embedding-0.6B"
DATASET_PATH="./data/wvs_sampled_sft/sampled_50k/train"

# Experiment Name: Removing top_k_routing_strategy enables "Soft Routing" (All experts)
EXP_NAME="cuma_wvs_50k_r64_a128_lr5e6_ablation_soft_routing"
SAVE_DIR="./checkpoints/${EXP_NAME}"

EVAL_ONLY=false
TARGET_MODULES="q_proj v_proj"

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
else
    if command -v nvidia-smi &> /dev/null; then
        NUM_GPUS=$(nvidia-smi -L | wc -l)
    else
        NUM_GPUS=1
    fi
fi

ZERO_STAGE=0
echo "Launching on $NUM_GPUS GPUs with DeepSpeed ZeRO-$ZERO_STAGE"
DEEPSPEED_CONFIG="scripts/ds_config_zero${ZERO_STAGE}.json"

TOTAL_BATCH_SIZE=16
PER_DEVICE_BATCH_SIZE=16
GRAD_ACC=$((TOTAL_BATCH_SIZE / (PER_DEVICE_BATCH_SIZE * NUM_GPUS)))

if [ "$EVAL_ONLY" = false ]; then
    echo "Starting training..."
    accelerate launch \
        --num_processes $NUM_GPUS \
        --num_machines 1 \
        --mixed_precision bf16 \
        --dynamo_backend no \
        --use_deepspeed \
        --deepspeed_config_file "${DEEPSPEED_CONFIG}" \
        --zero_stage $ZERO_STAGE \
        train/train_cuma_sft.py \
        --model "${MODEL_PATH}" \
        --embedding_model "${EMBEDDING_MODEL_PATH}" \
        --dataset_path "${DATASET_PATH}" \
        --save_dir "${SAVE_DIR}" \
        --bf16 \
        --num_epochs 3 \
        --batch_size ${PER_DEVICE_BATCH_SIZE} \
        --accumulation_steps ${GRAD_ACC} \
        --lr 5e-6 \
        --demographic_lr 5e-6 \
        --num_encoder_proj_mlp_layers 0 \
        --weight_decay 0.01 \
        --label_smoothing 0.0 \
        --max_grad_norm 1.0 \
        --warmup_steps 0 \
        --schedule_name constant \
        --max_length 1024 \
        --num_workers 4 \
        --lora_r 64 \
        --lora_alpha 128 \
        --dropout 0.0 \
        --num_experts 8 \
        --top_k 2 \
        --use_hydra_lora \
        --target_modules "${TARGET_MODULES}" \
        --share_router_for_qkv \
        --share_router_for_w_i \
        --demographic_embed_dim 1024 \
        --router_hidden_dim 256 \
        --num_router_mlp_layers 2 \
        --lambda_auxiliary 0.01 \
        --use_load_balancing_loss \
        --bf16 \
        --logging_steps 50 \
        --save_steps 999999 \
        --seed 42 \
        --use_demographic_system_prompt
fi

DATASET="./data/wvs_sampled_sft/sampled_50k/test"
PEFT_MODEL_PATH=${SAVE_DIR}

echo "Starting evaluation..."
accelerate launch \
    --num_processes $NUM_GPUS \
    --num_machines 1 \
    --mixed_precision bf16 \
    --dynamo_backend no \
    evaluate/evaluate_wvs.py \
    --model ${MODEL_PATH} \
    --embedding_model ${EMBEDDING_MODEL_PATH} \
    --dataset ${DATASET} \
    --eval_batch_size 32 \
    --bf16 \
    --seed 42 \
    --output_dir ./eval_results/${EXP_NAME} \
    --peft_model_path ${PEFT_MODEL_PATH} \
    --adapter_type cuma \
    --use_demographic_system_prompt
