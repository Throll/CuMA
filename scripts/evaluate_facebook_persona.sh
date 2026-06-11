#!/usr/bin/env bash

# Evaluate Facebook Persona Prompting
export PYTHONPATH="."
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_IB_DISABLE=1

# MODEL_PATH="../MyModels/Qwen3-8B"
MODEL_PATH="../MyModels/Llama-3.1-8B-Instruct"
DATASET="/root/autodl-tmp/data/facebook_sampled/test"
EXP_NAME="facebook_persona_8b"

accelerate launch \
  --num_processes 2 \
  --num_machines 1 \
  --mixed_precision bf16 \
  --dynamo_backend no \
  evaluate/evaluate_facebook.py \
  --model ${MODEL_PATH} \
  --dataset ${DATASET} \
  --mode persona \
  --few_shot 0 \
  --eval_batch_size 8 \
  --bf16 \
  --seed 42 \
  --output_dir ./eval_results/${EXP_NAME} \
  --adapter_type base
