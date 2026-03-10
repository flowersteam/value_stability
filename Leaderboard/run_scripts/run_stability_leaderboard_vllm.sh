#!/bin/bash
#SBATCH -A imi@h100
#SBATCH -C h100
#SBATCH --gres=gpu:4
#SBATCH --time=12:59:59
#SBATCH --cpus-per-task=24
#SBATCH -o slurm_logs/vllm_leaderboard_%A.client.log
#SBATCH -e slurm_logs/vllm_leaderboard_%A.client.log
#SBATCH -J vllm_eval
##SBATCH --qos=qos_gpu_h100-dev

start_time=$(date +%s)

# Server
##################

# launch server
source /linkhome/rech/genini01/utu57ed/.bashrc
module purge
module load arch/h100
module load python/3.12.2 || module load python/3.12.8 || module load python/3.12.7
conda activate vllm_new

echo "Launching server"

SERVERLOG="slurm_logs/vllm_leaderboard_$SLURM_JOB_ID.server.log"
echo -e "\e[32mServer log : $SERVERLOG\e[0m"


model_config=$1

model_config_path="./models/leaderboard_configs/$model_config.json"

echo "Model config path: $model_config_path"

# check if model config exists
if [ ! -f "$model_config_path" ]; then
  echo "Model config path doesn't exist: $model_config_path"
  exit 1
fi


model_path=$(python -c "import json; print(json.load(open('$model_config_path'))['model_path'])")
echo "Model path: $model_path"

# check if model exists
if [ ! -e "$model_path" ]; then
  echo "Model path doesn't exist: $model_path"
  exit 1
fi


#########

# for llama first launch vllm server with
#export VLLM_CONFIGURE_LOGGING=0 # no logging
#export VLLM_USE_V1=1
#export VLLM_LOGGING_LEVEL=DEBUG # no logging
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

#    --max-model-len 12000 \
#    --max-model-len 30000 \

vllm serve $model_path \
    --max-model-len 30000 \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.90 \
    --served-model-name "model" \
    --trust-remote-code \
    --dtype auto \
    --enable-prefix-caching  &> "$SERVERLOG" &

# wait for server to load
echo "Loading server"
start_time=$(date +%s)
while true; do
  # Attempt to connect
  response=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/v1/models)

  # check every 15 secs if server is loaded
  if [ "$response" -eq 000 ]; then
    echo "vllm server not loaded. Retrying in 15 seconds..."
    sleep 15
  else
    echo "vllm server running successfully!"
    break
  fi

  # 30 min timeout
  current_time=$(date +%s)
  elapsed=$((current_time - start_time))

  if [ "$elapsed" -ge 1800 ]; then
    echo "Timeout: vllm server failed to start within 30 minutes."
    exit 1
  fi
done


# Clients
##################
echo "Launching clients"
module purge
module load python/3.11.5
conda activate llm_stability_new

# launch clients


# subshell
(
for i in {0..8};
do
  CLIENTLOG="slurm_logs/vllm_leaderboard_$SLURM_JOB_ID.client_$i.log"
  echo -e "\e[32mClient log : $CLIENTLOG\e[0m"
  SLURM_ARRAY_TASK_ID=$i bash Leaderboard/run_scripts/run_stability_leaderboard.sh $model_config &> $CLIENTLOG &
done

wait
echo "Clients done"
)

# compute elapsed time
end_time=$(date +%s)
# Calculate the difference
duration=$(( end_time - start_time ))

# Calculate hours, minutes, and seconds
hours=$(( duration / 3600 ))
minutes=$(( (duration % 3600) / 60 ))
seconds=$(( duration % 60 ))

# Format the output to hh:mm:ss
printf -v elapsed_time "%02d:%02d:%02d" $hours $minutes $seconds