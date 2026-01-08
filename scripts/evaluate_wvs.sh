#!/usr/bin/env bash

# Evaluate vanilla baseline
# export CUDA_VISIBLE_DEVICES=1,2
export PYTHONPATH="."
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_IB_DISABLE=1

MODEL_PATH="../MyModels/Qwen3-8B"
# MODEL_PATH="../MyModels/Llama-3.1-8B-Instruct"
EMBEDDING_MODEL_PATH="../MyModels/Qwen3-Embedding-0.6B"
# EXP_NAME="persona_prompting_llama3.1_8b"
# EXP_NAME="persona_prompting"
EXP_NAME="persona_prompting_8b"
DATASET="./data/wvs_sampled_sft/sampled_10k/test"

accelerate launch \
  --num_processes 2 \
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
  --peft_model_path "" \
  --adapter_type base \
  --use_demographic_system_prompt

# For cultural adapter evaluation
# accelerate launch \
#   --num_processes 2 \
#   evaluate/evaluate_wvs.py \
#   --model ${MODEL_PATH} \
#   --dataset ${DATASET} \
#   --embedding_model ${EMBEDDING_MODEL_PATH} \
#   --eval_batch_size 32 \
#   --bf16 \
#   --seed 42 \
#   --output_dir ./eval_results/${EXP_NAME} \
#   --peft_model_path ${PEFT_MODEL_PATH} \
#   --adapter_type cultural
