#!/bin/bash

for i in {0..9};
do
  SLURM_ARRAY_TASK_ID=$i bash Leaderboard/run_scripts/run_stability_leaderboard.sh gpt-3.5-turbo-0125
done

for i in {0..9};
do
  SLURM_ARRAY_TASK_ID=$i bash Leaderboard/run_scripts/run_stability_leaderboard.sh gpt-4o-0513
done


sed -i "s/^#SBATCH --gres=.*/#SBATCH --gres=gpu:1/" Leaderboard/run_scripts/run_stability_leaderboard.sh
sbatch Leaderboard/run_scripts/run_stability_leaderboard.sh phi-3-mini-128k-instruct
sbatch Leaderboard/run_scripts/run_stability_leaderboard.sh phi-3-medium-128k-instruct
sbatch Leaderboard/run_scripts/run_stability_leaderboard.sh llama_3_8b_instruct
sbatch Leaderboard/run_scripts/run_stability_leaderboard.sh Qwen2-7B-Instruct
sbatch Leaderboard/run_scripts/run_stability_leaderboard.sh Mistral-7B-Instruct-v0.3
sbatch Leaderboard/run_scripts/run_stability_leaderboard.sh Mistral-7B-Instruct-v0.2
sbatch Leaderboard/run_scripts/run_stability_leaderboard.sh Mistral-7B-Instruct-v0.1


sed -i "s/^#SBATCH --gres=.*/#SBATCH --gres=gpu:2/" Leaderboard/run_scripts/run_stability_leaderboard.sh
sbatch Leaderboard/run_scripts/run_stability_leaderboard.sh Mixtral-8x7B-Instruct-v0.1
sbatch Leaderboard/run_scripts/run_stability_leaderboard.sh llama_3_70b_instruct
sbatch Leaderboard/run_scripts/run_stability_leaderboard.sh phi-3-medium-128k-instruct # chunk_0

sed -i "s/^#SBATCH --gres=.*/#SBATCH --gres=gpu:3/" Leaderboard/run_scripts/run_stability_leaderboard.sh
sbatch Leaderboard/run_scripts/run_stability_leaderboard.sh command_r_plus
sbatch Leaderboard/run_scripts/run_stability_leaderboard.sh Qwen2-72B-Instruct

sed -i "s/^#SBATCH --gres=.*/#SBATCH --gres=gpu:4/" Leaderboard/run_scripts/run_stability_leaderboard.sh
sbatch Leaderboard/run_scripts/run_stability_leaderboard.sh Mixtral-8x22B-Instruct-v0.1

sbatch Leaderboard/run_scripts/run_stability_leaderboard.sh command_r_plus # chunk_0
sbatch Leaderboard/run_scripts/run_stability_leaderboard.sh Qwen2-72B-Instruct # chunk_0