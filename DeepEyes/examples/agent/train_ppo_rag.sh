set -x

# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
DATA_DIR=/cpfs/user/fengyuan/verl_data/nq_hotpot
# DATA_DIR=./data/gsm8k

PROJECT_NAME="agent_ppo_debug"
EXPERIMENT_NAME="limit_single_response_length"

# # set -x
# export VLLM_ATTENTION_BACKEND=XFORMERS # vllm + qwen2-7b with flash_attn has some issues

REF_MODEL_PATH=/cpfs/user/fengyuan/backbone/qwen25/Qwen2.5-7B

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files=${DATA_DIR}/train.parquet \
    data.val_files=${DATA_DIR}/test.parquet \
    data.train_batch_size=512 \
    data.max_prompt_length=4096 \
    data.max_response_length=8192 \
    algorithm.adv_estimator=gae \
    algorithm.kl_ctrl.kl_coef=0.0 \
    algorithm.lam=1.0 \
    actor_rollout_ref.model.path=${REF_MODEL_PATH} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.agent.activate_agent=True \
    actor_rollout_ref.rollout.agent.tool_name_key=env_name \
    actor_rollout_ref.rollout.agent.single_response_max_tokens=1024 \
    actor_rollout_ref.rollout.agent.single_obs_max_length=8192 \
    actor_rollout_ref.rollout.agent.max_turns=8 \
    actor_rollout_ref.rollout.agent.concurrent_workers=1 \
    actor_rollout_ref.rollout.agent.show_tqdm=True \
    critic.optim.lr=1e-5 \
    critic.model.path=${REF_MODEL_PATH} \
    critic.model.fsdp_config.param_offload=True \
    critic.model.fsdp_config.optimizer_offload=True \
    critic.ppo_micro_batch_size_per_gpu=1 \
    trainer.logger=['console','wandb'] \
    +trainer.val_before_train=False \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=${WORLD_SIZE} \
    trainer.save_freq=20 \
    trainer.test_freq=20 \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.total_epochs=15 2>&1 | tee ./logs/${EXPERIMENT_NAME}.log
