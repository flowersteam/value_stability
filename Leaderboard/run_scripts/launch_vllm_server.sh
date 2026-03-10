#!/bin/bash
#SBATCH -A imi@h100
#SBATCH -C h100
#SBATCH --gres=gpu:1
#SBATCH --time=01:59:59
#SBATCH --cpus-per-task=24
#SBATCH -o slurm_logs/vllm_leaderboard_%A.client.log
#SBATCH -e slurm_logs/vllm_leaderboard_%A.client.log
#SBATCH -J vllm_eval
##SBATCH --qos=qos_gpu_a100-dev
#SBATCH --qos=qos_gpu_h100-dev

start_time=$(date +%s)

# Server
##################
# launch server
source /linkhome/rech/genini01/utu57ed/.bashrc
module purge
module load arch/h100
module load python/3.12.2
#conda activate vllm_312
conda activate vllm_new

echo "Launching server"

model_config=$1
#model_config="gemma-3-27b-it_vllm_freeform"

# extract model path
model_config_path="./models/leaderboard_configs/$model_config.json"
model_path=$(python -c "import json; print(json.load(open('$model_config_path'))['model_path'])")

echo "Model config path: $model_config_path"
echo "Model path: $model_path"

#########

# for regular models
export VLLM_CONFIGURE_LOGGING=0 # no logging
export VLLM_USE_V1=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
vllm serve $model_path \
    --tensor-parallel-size 1 \
    --max-model-len 30000 \
    --gpu-memory-utilization 0.90 \
    --served-model-name "model" \
    --trust-remote-code \
    --dtype auto \
    --max-num-seqs 10 \
    --enable-prefix-caching


# for llama-4
#VLLM_DISABLE_COMPILE_CACHE=1 vllm serve $model_path \
#  --tensor-parallel-size 4 \
#  --served-model-name "model" \
#  --max-model-len 30000 \
#  --override-generation-config='{"attn_temperature_tuning": true}'

#vllm serve $model_path \
#    --tensor-parallel-size 4 \
#    --max-model-len 30000 \
#    --gpu-memory-utilization 0.90 \
#    --served-model-name "model" \
#    --trust-remote-code \
#    --dtype auto \
#    --max-num-seqs 10 \
#    --enable-prefix-caching


## mistral
#export VLLM_WORKER_MULTIPROC_METHOD=spawn
#vllm serve $model_path \
#    --tensor-parallel-size 4 \
#    --tokenizer_mode mistral \
#    --config_format mistral \
#    --load_format mistral \
#    --tool-call-parser mistral \
#    --max-model-len 12000 \
#    --gpu-memory-utilization 0.90 \
#    --served-model-name "model" \
#    --trust-remote-code \
#    --dtype auto \
#    --max-num-seqs 10 \
#    --enable-prefix-caching

#vllm serve mistralai/Mistral-Small-3.1-24B-Instruct-2503 --tokenizer_mode mistral --config_format mistral
#--load_format mistral --tool-call-parser mistral --enable-auto-tool-choice --limit_mm_per_prompt 'image=10' --tensor-parallel-size 2