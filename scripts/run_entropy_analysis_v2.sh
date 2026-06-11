#!/usr/bin/env bash
set -eu

CUDA_VISIBLE_DEVICES=0
source .venv/bin/activate
export PYTHONPATH="."

MODEL_PATH="/root/autodl-tmp/Qwen3-8B"
EMBEDDING_MODEL_PATH="/root/autodl-tmp/Qwen3-Embedding-0.6B"
DATASET="./data/wvs_sampled_sft/sampled_50k/test"
OUTPUT_DIR="eval_results/entropy"

mkdir -p $OUTPUT_DIR

# # 1. FFT (Full Fine-Tuning)
# echo "Calculating entropy for FFT..."
# python evaluate/calculate_entropy.py \
#     --model ${MODEL_PATH} \
#     --peft_model_path "../autodl-fs/checkpoints/fft_wvs_50k_zero2/" \
#     --dataset ${DATASET} \
#     --adapter_type base \
#     --output_name "FFT" \
#     --output_dir ${OUTPUT_DIR} \
#     --bf16

# # 2. P-Tuning v2
# echo "Calculating entropy for P-Tuning v2..."
# python evaluate/calculate_entropy.py \
#     --model ${MODEL_PATH} \
#     --peft_model_path "./checkpoints/pturning2_wvs_50k" \
#     --dataset ${DATASET} \
#     --adapter_type peft \
#     --output_name "P-Tuning_v2" \
#     --output_dir ${OUTPUT_DIR} \
#     --bf16

# # 3. LoRA (r=64)
# echo "Calculating entropy for LoRA (r=64)..."
# python evaluate/calculate_entropy.py \
#     --model ${MODEL_PATH} \
#     --peft_model_path "./checkpoints/lora_wvs_50k_r64" \
#     --dataset ${DATASET} \
#     --adapter_type lora \
#     --output_name "LoRA_r64" \
#     --output_dir ${OUTPUT_DIR} \
#     --bf16

# # 4. DoRA (r=64)
# echo "Calculating entropy for DoRA (r=64)..."
# python evaluate/calculate_entropy.py \
#     --model ${MODEL_PATH} \
#     --peft_model_path "./checkpoints/dora_wvs_50k_r64_lr5e6" \
#     --dataset ${DATASET} \
#     --adapter_type lora \
#     --output_name "DoRA_r64" \
#     --output_dir ${OUTPUT_DIR} \
#     --bf16

# # 5. MixLoRA
# echo "Calculating entropy for MixLoRA..."
# python evaluate/calculate_entropy.py \
#     --model ${MODEL_PATH} \
#     --peft_model_path "./checkpoints/mixlora_wvs_50k_qwen3_8b" \
#     --dataset ${DATASET} \
#     --adapter_type mixlora \
#     --output_name "MixLoRA" \
#     --output_dir ${OUTPUT_DIR} \
#     --bf16

# # 6. HydraLoRA
# echo "Calculating entropy for HydraLoRA..."
# python evaluate/calculate_entropy.py \
#     --model ${MODEL_PATH} \
#     --peft_model_path "./checkpoints/hydralora_wvs_50k_qwen3_8b" \
#     --dataset ${DATASET} \
#     --adapter_type hydralora \
#     --output_name "HydraLoRA" \
#     --output_dir ${OUTPUT_DIR} \
#     --bf16

# # 7. CuMA (r=64)
# echo "Calculating entropy for CuMA (r=64)..."
# python evaluate/calculate_entropy.py \
#     --model ${MODEL_PATH} \
#     --embedding_model ${EMBEDDING_MODEL_PATH} \
#     --peft_model_path "./checkpoints/cuma_wvs_50k_r64_a128_lr5e6" \
#     --dataset ${DATASET} \
#     --adapter_type cuma \
#     --output_name "CuMA_r64" \
#     --output_dir ${OUTPUT_DIR} \
#     --use_demographic_system_prompt \
#     --bf16

# 8. Plot results
echo "Generating comparison plot..."
python utils/plot_entropy.py --input_dir ${OUTPUT_DIR} --output_file "${OUTPUT_DIR}/entropy_comparison_v2.png"
