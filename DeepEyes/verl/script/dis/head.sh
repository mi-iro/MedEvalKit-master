source /volume/pt-train/miniconda3/bin/activate
conda activate DeepEyes
cd /volume/pt-train/users/txie/misc/M3/DeepEyes/verl/script/dis


echo ${MASTER_ADDR}
echo $PET_NODE_RANK
export NCCL_TIMEOUT=36000


ray stop --force
# unset RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES
CURRENT_NODE_IP=$(hostname -i)
echo "Start Ray head node"
MY_IP=$(hostname -i)
echo $MY_IP > ./ray_head_ip
ray start --head --dashboard-host=0.0.0.0 --num-gpus 8

sleep 60 # wait for ray to start

sh /volume/pt-train/users/txie/misc/M3/DeepEyes/verl/script/dis/submit.sh
# ray status
# sleep 10
