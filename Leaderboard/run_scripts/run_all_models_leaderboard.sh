#!/bin/bash
#for i in 0 6 7
#do
#  SLURM_ARRAY_TASK_ID=$i bash Leaderboard/run_scripts/run_stability_leaderboard.sh DeepSeek-R1 &
#done
#wait
#for i in 3 4 5;
#do
#  SLURM_ARRAY_TASK_ID=$i bash Leaderboard/run_scripts/run_stability_leaderboard.sh DeepSeek-R1 &
#done
#wait
#for i in 1 2 8;
#do
#  SLURM_ARRAY_TASK_ID=$i bash Leaderboard/run_scripts/run_stability_leaderboard.sh DeepSeek-R1 &
#done
#wait
#exit

# ds models
#sbatch Leaderboard/run_scripts/run_stability_leaderboard.sh DeepSeek-V3-0324
#sbatch Leaderboard/run_scripts/run_stability_leaderboard.sh DeepSeek-V3-0324_user

# mistral small
#sbatch Leaderboard/run_scripts/run_stability_leaderboard_vllm_mistral.sh Mistral-Small-3.1-24B-Instruct-2503

#sbatch Leaderboard/run_scripts/run_stability_leaderboard_vllm_qwen3.sh Qwen3-235B-A22B-FP8


# vllm models
#sbatch Leaderboard/run_scripts/run_stability_leaderboard_vllm.sh Llama-4-Scout-17B-16E-Instruct
#sbatch Leaderboard/run_scripts/run_stability_leaderboard_vllm.sh Llama-3.3-70B-Instruct
#sbatch Leaderboard/run_scripts/run_stability_leaderboard_vllm.sh Llama-3.1-70B-Instruct
#sbatch Leaderboard/run_scripts/run_stability_leaderboard_vllm.sh Llama-3.1-Nemotron-70B-Instruct
#sbatch Leaderboard/run_scripts/run_stability_leaderboard_vllm.sh Llama-3.1-8B-Instruct
#sbatch Leaderboard/run_scripts/run_stability_leaderboard_vllm.sh Llama-3.2-3B-Instruct
#sbatch Leaderboard/run_scripts/run_stability_leaderboard_vllm.sh Llama-3.2-1B-Instruct

#sbatch Leaderboard/run_scripts/run_stability_leaderboard_vllm.sh reka-flash-3
#sbatch Leaderboard/run_scripts/run_stability_leaderboard_vllm.sh gemma-3-27b-it

#sbatch Leaderboard/run_scripts/run_stability_leaderboard_vllm.sh Mistral-Large-Instruct-2411
#sbatch Leaderboard/run_scripts/run_stability_leaderboard_vllm.sh Mistral-Large-Instruct-2407
#sbatch Leaderboard/run_scripts/run_stability_leaderboard_vllm.sh Mistral-Nemo-Instruct-2407
#sbatch Leaderboard/run_scripts/run_stability_leaderboard_vllm.sh QwQ-32B
#sbatch Leaderboard/run_scripts/run_stability_leaderboard_vllm.sh Qwen2.5-VL-72B-Instruct
sbatch Leaderboard/run_scripts/run_stability_leaderboard_vllm.sh Qwen2.5-VL-32B-Instruct
#sbatch Leaderboard/run_scripts/run_stability_leaderboard_vllm.sh Qwen2.5-VL-7B-Instruct
#sbatch Leaderboard/run_scripts/run_stability_leaderboard_vllm.sh Qwen2.5-VL-3B-Instruct
#sbatch Leaderboard/run_scripts/run_stability_leaderboard_vllm.sh Qwen2.5-72B-Instruct # skip?
#sbatch Leaderboard/run_scripts/run_stability_leaderboard_vllm.sh Qwen2.5-32B-Instruct # skip?
#sbatch Leaderboard/run_scripts/run_stability_leaderboard_vllm.sh Qwen2.5-14B-Instruct-1M

#sbatch Leaderboard/run_scripts/run_stability_leaderboard_vllm.sh phi-4 # svs # not enough context for CoT

#sbatch Leaderboard/run_scripts/run_stability_leaderboard_vllm.sh Dracarys2-72B-Instruct
#sbatch Leaderboard/run_scripts/run_stability_leaderboard_vllm.sh Nautilus-70B-v0.1
#sbatch Leaderboard/run_scripts/run_stability_leaderboard_vllm.sh Cydonia-22B-v1.2
#sbatch Leaderboard/run_scripts/run_stability_leaderboard_vllm.sh Ministrations-8B-v1

# set reasoning mode
#huggingface-cli download nvidia/Llama-3_3-Nemotron-Super-49B-v1 # set reasoning mode
#huggingface-cli download nvidia/Llama-3.1-Nemotron-Nano-8B-v1 # set reasoning mode


# DeepSeek
#SLURM_ARRAY_TASK_ID=0 bash Leaderboard/run_scripts/run_stability_leaderboard.sh DeepSeek-V3-0324 #
#SLURM_ARRAY_TASK_ID=6 bash Leaderboard/run_scripts/run_stability_leaderboard.sh DeepSeek-V3-0324 #
#SLURM_ARRAY_TASK_ID=7 bash Leaderboard/run_scripts/run_stability_leaderboard.sh DeepSeek-V3-0324 #
#SLURM_ARRAY_TASK_ID=5 bash Leaderboard/run_scripts/run_stability_leaderboard.sh DeepSeek-V3-0324 #
#SLURM_ARRAY_TASK_ID=4 bash Leaderboard/run_scripts/run_stability_leaderboard.sh DeepSeek-V3-0324 #
#SLURM_ARRAY_TASK_ID=3 bash Leaderboard/run_scripts/run_stability_leaderboard.sh DeepSeek-V3-0324 # R
#SLURM_ARRAY_TASK_ID=2 bash Leaderboard/run_scripts/run_stability_leaderboard.sh DeepSeek-V3-0324 # R
#SLURM_ARRAY_TASK_ID=1 bash Leaderboard/run_scripts/run_stability_leaderboard.sh DeepSeek-V3-0324 # R
#SLURM_ARRAY_TASK_ID=8 bash Leaderboard/run_scripts/run_stability_leaderboard.sh DeepSeek-V3-0324 # R

#for i in {1..8};
#do
#  SLURM_ARRAY_TASK_ID=$i bash Leaderboard/run_scripts/run_stability_leaderboard.sh llama_3.1_405b_instruct
#done
#
#exit
#
#for i in {0..8};
#do
#  SLURM_ARRAY_TASK_ID=$i bash Leaderboard/run_scripts/run_stability_leaderboard.sh gpt-4o-mini-2024-07-18
#done
#for i in {0..8};
#do
#  SLURM_ARRAY_TASK_ID=$i bash Leaderboard/run_scripts/run_stability_leaderboard.sh gpt-3.5-turbo-0125
#done
#
#for i in {0..8};
#do
#  SLURM_ARRAY_TASK_ID=$i bash Leaderboard/run_scripts/run_stability_leaderboard.sh gpt-4o-0513
#done
#
#

#SLURM_ARRAY_TASK_ID=0 bash Leaderboard/run_scripts/run_stability_leaderboard.sh gemma-2-27b-it


#sed -i "s/^#SBATCH --gres=.*/#SBATCH --gres=gpu:1/" Leaderboard/run_scripts/run_stability_leaderboard.sh
#sbatch Leaderboard/run_scripts/run_stability_leaderboard.sh gemma-3-27b-it
#sbatch Leaderboard/run_scripts/run_stability_leaderboard.sh Dolphin3.0-Mistral-24B
#sbatch Leaderboard/run_scripts/run_stability_leaderboard.sh Qwen2.5-14B-Instruct-1M
#sbatch Leaderboard/run_scripts/run_stability_leaderboard.sh phi-4
#sbatch Leaderboard/run_scripts/run_stability_leaderboard.sh Falcon3-10B-Instruct

#sbatch Leaderboard/run_scripts/run_stability_leaderboard.sh OLMo-2-1124-7B-Instruct
#sbatch Leaderboard/run_scripts/run_stability_leaderboard.sh OLMo-2-1124-13B-Instruct

#sbatch Leaderboard/run_scripts/run_stability_leaderboard.sh Ministral-8B-Instruct-2410
#sbatch Leaderboard/run_scripts/run_stability_leaderboard.sh Ministrations-8B-v1

#sbatch Leaderboard/run_scripts/run_stability_leaderboard.sh UnslopNemo-12B-v4.1
#sbatch Leaderboard/run_scripts/run_stability_leaderboard.sh Cydonia-22B-v1.2
#sbatch Leaderboard/run_scripts/run_stability_leaderboard.sh Behemoth-123B-v1.1

#sbatch Leaderboard/run_scripts/run_stability_leaderboard.sh hermes_3_llama_3.1_8b_instruct
#sbatch Leaderboard/run_scripts/run_stability_leaderboard.sh gemma-2-2b-it
#sbatch Leaderboard/run_scripts/run_stability_leaderboard.sh gemma-2-9b-it
#sbatch Leaderboard/run_scripts/run_stability_leaderboard.sh gemma-2-27b-it
#sbatch Leaderboard/run_scripts/run_stability_leaderboard.sh Mistral-Nemo-Instruct-2407
#sbatch Leaderboard/run_scripts/run_stability_leaderboard.sh phi-3.5-mini-instruct
#sbatch Leaderboard/run_scripts/run_stability_leaderboard.sh Mistral-Small-Instruct-2409

#sbatch Leaderboard/run_scripts/run_stability_leaderboard.sh Qwen2.5-0.5B-Instruct
#sbatch Leaderboard/run_scripts/run_stability_leaderboard.sh Qwen2.5-7B-Instruct
#sbatch Leaderboard/run_scripts/run_stability_leaderboard.sh Qwen2.5-32B-Instruct

#sbatch Leaderboard/run_scripts/run_stability_leaderboard.sh phi-3-mini-128k-instruct
#sbatch Leaderboard/run_scripts/run_stability_leaderboard.sh phi-3-medium-128k-instruct
#sbatch Leaderboard/run_scripts/run_stability_leaderboard.sh llama_3_8b_instruct
#sbatch Leaderboard/run_scripts/run_stability_leaderboard.sh llama_3.1_8b_instruct
#sbatch Leaderboard/run_scripts/run_stability_leaderboard.sh Qwen2-7B-Instruct
#sbatch Leaderboard/run_scripts/run_stability_leaderboard.sh Mistral-7B-Instruct-v0.3
#sbatch Leaderboard/run_scripts/run_stability_leaderboard.sh Mistral-7B-Instruct-v0.2
#sbatch Leaderboard/run_scripts/run_stability_leaderboard.sh Mistral-7B-Instruct-v0.1
#sbatch Leaderboard/run_scripts/run_stability_leaderboard.sh llama_3.2_1b_instruct

#sed -i "s/^#SBATCH --gres=.*/#SBATCH --gres=gpu:2/" Leaderboard/run_scripts/run_stability_leaderboard.sh
#sbatch Leaderboard/run_scripts/run_stability_leaderboard.sh dolphin-2.9.2-qwen2-72b
#sbatch Leaderboard/run_scripts/run_stability_leaderboard.sh Dracarys-72B-Instruct
#sbatch Leaderboard/run_scripts/run_stability_leaderboard.sh Llama-3.1-Tulu-3-70B-DPO
#sbatch Leaderboard/run_scripts/run_stability_leaderboard.sh Nautilus-70B-v0.1
#sbatch Leaderboard/run_scripts/run_stability_leaderboard.sh phi-3.5-MoE-instruct
#sbatch Leaderboard/run_scripts/run_stability_leaderboard.sh Mixtral-8x7B-Instruct-v0.1
#sbatch Leaderboard/run_scripts/run_stability_leaderboard.sh llama_3_70b_instruct
#sbatch Leaderboard/run_scripts/run_stability_leaderboard.sh llama_3.1_70b_instruct
#sbatch Leaderboard/run_scripts/run_stability_leaderboard.sh llama_3.3_70b_instruct
#sbatch Leaderboard/run_scripts/run_stability_leaderboard.sh phi-3-medium-128k-instruct # chunk_0
#sbatch Leaderboard/run_scripts/run_stability_leaderboard.sh hermes_3_llama_3.1_70b
#sbatch Leaderboard/run_scripts/run_stability_leaderboard.sh Llama-3.1-Centaur-70B

#
#sed -i "s/^#SBATCH --gres=.*/#SBATCH --gres=gpu:3/" Leaderboard/run_scripts/run_stability_leaderboard.sh
#sbatch Leaderboard/run_scripts/run_stability_leaderboard.sh command_r_plus
#sbatch Leaderboard/run_scripts/run_stability_leaderboard.sh Qwen2-72B-Instruct
#sbatch Leaderboard/run_scripts/run_stability_leaderboard.sh Mistral-Large-Instruct-2407
#sbatch Leaderboard/run_scripts/run_stability_leaderboard.sh Qwen2.5-72B-Instruct
#
#sed -i "s/^#SBATCH --gres=.*/#SBATCH --gres=gpu:4/" Leaderboard/run_scripts/run_stability_leaderboard.sh
#sbatch Leaderboard/run_scripts/run_stability_leaderboard.sh Mistral-Large-Instruct-2411
#sbatch Leaderboard/run_scripts/run_stability_leaderboard.sh Mixtral-8x22B-Instruct-v0.1
#
#sbatch Leaderboard/run_scripts/run_stability_leaderboard.sh command_r_plus # chunk_0
#sbatch Leaderboard/run_scripts/run_stability_leaderboard.sh Qwen2-72B-Instruct # chunk_0

#sbatch Leaderboard/run_scripts/run_stability_leaderboard.sh Mistral-Large-Instruct-2407 # chunks-0-3 svs

# 4 bits
#sbatch Leaderboard/run_scripts/run_stability_leaderboard.sh llama_3.1_405b_instruct_4bit
