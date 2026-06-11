#!/usr/bin/env bash
set -eu

CUDA_VISIBLE_DEVICES=0
source .venv/bin/activate
export PYTHONPATH="."

MODEL_PATH="../MyModels/Qwen3-8B"
EMBEDDING_MODEL_PATH="../MyModels/Qwen3-Embedding-0.6B"
DATASET="../autodl-tmp/data/prism_sampled_sft_2000/test"
OUTPUT_DIR="eval_results/distinct_prism"
BATCH_SIZE=32

mkdir -p $OUTPUT_DIR

# 2. P-Tuning v2
echo "Calculating Distinct-N for P-Tuning v2..."
python evaluate/calculate_distinct.py \
    --model ${MODEL_PATH} \
    --peft_model_path "./checkpoints/pturning2_wvs_50k" \
    --dataset ${DATASET} \
    --adapter_type peft \
    --output_name "P-Tuning_v2" \
    --output_dir ${OUTPUT_DIR} \
    --eval_batch_size ${BATCH_SIZE} \
    --bf16

# 3. LoRA (r=64)
echo "Calculating Distinct-N for LoRA (r=64)..."
python evaluate/calculate_distinct.py \
    --model ${MODEL_PATH} \
    --peft_model_path "./checkpoints/lora_wvs_50k_r64" \
    --dataset ${DATASET} \
    --adapter_type lora \
    --output_name "LoRA_r64" \
    --output_dir ${OUTPUT_DIR} \
    --eval_batch_size ${BATCH_SIZE} \
    --bf16

# 4. DoRA (r=64)
echo "Calculating Distinct-N for DoRA (r=64)..."
python evaluate/calculate_distinct.py \
    --model ${MODEL_PATH} \
    --peft_model_path "./checkpoints/dora_wvs_50k_r64_lr5e6" \
    --dataset ${DATASET} \
    --adapter_type dora \
    --output_name "DoRA_r64" \
    --output_dir ${OUTPUT_DIR} \
    --eval_batch_size ${BATCH_SIZE} \
    --bf16

# 5. MixLoRA (r=64)
echo "Calculating Distinct-N for MixLoRA (r=64)..."
python evaluate/calculate_distinct.py \
    --model ${MODEL_PATH} \
    --peft_model_path "./checkpoints/mixlora_wvs_50k_qwen3_8b" \
    --dataset ${DATASET} \
    --adapter_type mixlora \
    --output_name "MixLoRA_r64" \
    --output_dir ${OUTPUT_DIR} \
    --eval_batch_size ${BATCH_SIZE} \
    --bf16

# 6. HydraLoRA (r=64)
echo "Calculating Distinct-N for HydraLoRA (r=64)..."
python evaluate/calculate_distinct.py \
    --model ${MODEL_PATH} \
    --peft_model_path "./checkpoints/hydralora_wvs_50k_qwen3_8b" \
    --dataset ${DATASET} \
    --adapter_type hydralora \
    --output_name "HydraLoRA_r64" \
    --output_dir ${OUTPUT_DIR} \
    --eval_batch_size ${BATCH_SIZE} \
    --bf16

# 7. CuMA (r=64)
echo "Calculating Distinct-N for CuMA (r=64)..."
python evaluate/calculate_distinct.py \
    --model ${MODEL_PATH} \
    --peft_model_path "./checkpoints/cuma_wvs_50k_r64_a128_lr5e6" \
    --embedding_model ${EMBEDDING_MODEL_PATH} \
    --dataset ${DATASET} \
    --adapter_type cuma \
    --output_name "CuMA_r64" \
    --output_dir ${OUTPUT_DIR} \
    --eval_batch_size ${BATCH_SIZE} \
    --bf16
