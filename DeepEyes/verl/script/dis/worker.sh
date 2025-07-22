source /volume/pt-train/miniconda3/bin/activate
conda activate DeepEyes
cd /volume/pt-train/users/txie/misc/M3/DeepEyes/verl/script/dis


echo ${MASTER_ADDR}
echo $PET_NODE_RANK
export NCCL_TIMEOUT=36000


ray stop --force
# unset RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES
CURRENT_NODE_IP=$(hostname -i)

while [ ! -f ./ray_head_ip ]; do sleep 2; done
HEAD_IP=$(cat ./ray_head_ip)
echo "Start Ray worker node to ${HEAD_IP}"
ray start --address ${HEAD_IP}:6379  --num-gpus 8
    