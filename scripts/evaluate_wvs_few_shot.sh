#!/usr/bin/env bash

# Evaluate few-shot persona prompting
export PYTHONPATH="."
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_IB_DISABLE=1

MODEL_PATH="../MyModels/Qwen3-8B"
# MODEL_PATH="../MyModels/Llama-3.1-8B-Instruct"
EXP_NAME="few_shot_3_persona_prompting_8b_qwen3"
DATASET="/root/autodl-tmp/data/wvs_sampled_sft/sampled_10k/test"
TRAIN_DATASET="/root/autodl-tmp/data/wvs_sampled_sft/sampled_10k/train"

accelerate launch \
  --num_processes 2 \
  --num_machines 1 \
  --mixed_precision bf16 \
  --dynamo_backend no \
  evaluate/evaluate_wvs_few_shot.py \
  --model ${MODEL_PATH} \
  --dataset ${DATASET} \
  --train_dataset ${TRAIN_DATASET} \
  --few_shot 3 \
  --eval_batch_size 8 \
  --bf16 \
  --seed 42 \
  --output_dir ./eval_results/${EXP_NAME} \
  --peft_model_path "" \
  --adapter_type base \
  --use_demographic_system_prompt
