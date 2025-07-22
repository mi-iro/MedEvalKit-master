echo ${MASTER_ADDR}
echo $PET_NODE_RANK
export NCCL_TIMEOUT=36000


ray stop --force
# unset RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES
CURRENT_NODE_IP=$(hostname -i)
echo "current IP: $CURRENT_NODE_IP"

if [ $PET_NODE_RANK -eq 0 ]; then
    # Start Ray head node
    echo "Start Ray head node"
    MY_IP=$(hostname -i)
    echo $MY_IP > ./ray_head_ip
    ray start --head --dashboard-host=0.0.0.0 --num-gpus 8
    sleep 120

    
else
    sleep 60
    sleep ${PET_NODE_RANK}0
    while [ ! -f ./ray_head_ip ]; do sleep 2; done
    HEAD_IP=$(cat ./ray_head_ip)
    echo "Start Ray worker node to ${HEAD_IP}"
    ray start --address ${HEAD_IP}:6379  --num-gpus 8
    sleep 60
fi
