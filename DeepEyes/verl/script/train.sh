# your wandb access key here...
export WANDB_API_KEY=debf45c8d5066727456db660e57400d78b751446
export WANDB_MODE=offline
# the IP and port for your Qwen-2.5-72B-Instruct vllm serving


# umber of training nodes
# export WORLD_SIZE=4

# config for 7B
bash final_merged_v1v8_thinklite.sh

# # config for 32B
# bash examples/agent/final_merged_v1v8_thinklite_32b.sh