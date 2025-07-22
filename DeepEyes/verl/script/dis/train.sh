# your wandb access key here...
export WANDB_API_KEY=debf45c8d5066727456db660e57400d78b751446
export WANDB_MODE=offline
# the IP and port for your Qwen-2.5-72B-Instruct vllm serving
export LLM_AS_A_JUDGE_BASE="http://172.16.43.156:18900/v1"

# umber of training nodes
export WORLD_SIZE=4

# config for 7B
bash /volume/pt-train/users/txie/misc/M3/DeepEyes/verl/script/dis/M3.sh

# # config for 32B
# bash examples/agent/final_merged_v1v8_thinklite_32b.sh