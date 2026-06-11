#!/usr/bin/env bash
set -eu

source .venv/bin/activate
export PYTHONPATH="."
export TOKENIZERS_PARALLELISM=false

MODEL_PATH="../MyModels/Qwen3-8B"
DATASET_PATH="./data/wvs_sampled_sft/sampled_50k/train"

EXP_NAME="prefix_wvs_50k"
SAVE_DIR="./checkpoints/${EXP_NAME}"

EVAL_ONLY=false

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
else
    if command -v nvidia-smi &> /dev/null; then
        NUM_GPUS=$(nvidia-smi -L | wc -l)
    else
        NUM_GPUS=1
    fi
fi

ZERO_STAGE=2
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
        train/train_prefix_sft.py \
        --model "${MODEL_PATH}" \
        --dataset_path "${DATASET_PATH}" \
        --save_dir "${SAVE_DIR}" \
        --bf16 \
        --num_epochs 3 \
        --batch_size ${PER_DEVICE_BATCH_SIZE} \
        --accumulation_steps ${GRAD_ACC} \
        --lr 2e-5 \
        --weight_decay 0.01 \
        --warmup_steps 0 \
        --schedule_name constant \
        --max_length 1024 \
        --num_workers 4 \
        --logging_steps 50 \
        --save_steps 999999 \
        --seed 42 \
        --use_demographic_system_prompt \
        --dataset_type wvs \
        --num_virtual_tokens 32
        # --prefix_projection
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
    --dataset ${DATASET} \
    --eval_batch_size 32 \
    --bf16 \
    --seed 42 \
    --output_dir ./eval_results/${EXP_NAME} \
    --peft_model_path ${PEFT_MODEL_PATH} \
    --adapter_type prefix \
    --use_demographic_system_prompt
