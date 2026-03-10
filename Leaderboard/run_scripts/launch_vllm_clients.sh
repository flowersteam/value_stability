#!/bin/bash
#SBATCH -A imi@h100
#SBATCH -C h100
#SBATCH --gres=gpu:2
#SBATCH --time=01:59:59
#SBATCH --cpus-per-task=24
#SBATCH -o slurm_logs/vllm_leaderboard_%A.client.log
#SBATCH -e slurm_logs/vllm_leaderboard_%A.client.log
#SBATCH -J vllm_eval
##SBATCH --qos=qos_gpu_a100-dev
#SBATCH --qos=qos_gpu_h100-dev

# Clients
##################
echo "Launching clients"
module purge
module load python/3.11.5
conda activate llm_stability_311

model_config=$1
#model_config="gemma-3-27b-it_vllm_freeform"

#SLURM_ARRAY_TASK_ID=0 bash Leaderboard/run_scripts/run_stability_leaderboard.sh Llama-3.1-8B-Instruct
#SLURM_ARRAY_TASK_ID=0 bash Leaderboard/run_scripts/run_stability_leaderboard.sh Llama-4-Scout-17B-16E-Instruct
#SLURM_ARRAY_TASK_ID=0 bash Leaderboard/run_scripts/run_stability_leaderboard.sh Llama-4-Scout-17B-16E-Instruct
exit

curl http://localhost:8000/v1/completions -H "Content-Type: application/json" -d '{"model": "model", "prompt": "San Francisco is a", "max_tokens": 7, "temperature": 0 }'
#python -c 'from openai import OpenAI; print(OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1").completions.create(model="model", prompt="San Francisco is a"))'
#from openai import OpenAI; print(OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1").completions.create(model="model", prompt="San Francisco is a"))

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
