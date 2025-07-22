#!/bin/bash
# Set XFormers backend to avoid CUDA errors
# export VLLM_ATTENTION_BACKEND=XFORMERS
# Run 8K context length training
# export MODEL_PATH="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
# SCRIPT=$1

sleep 20 # wait for ray to start
ray status
sleep 10
if [ $PET_NODE_RANK -eq 0 ]; then
ray job submit --address="http://127.0.0.1:8265"  --runtime-env-json '{"env_vars": {"WANDB_API_KEY": "16b729e78da3bfd13cf6be5abc238842eee21e22", "HF_TOKEN": "debf45c8d5066727456db660e57400d78b751446"}}'  -- bash /volume/pt-train/users/txie/misc/M3/DeepEyes/verl/script/train.sh
    # bash ./scripts/train/${SCRIPT} --model $MODEL_PATH
fi