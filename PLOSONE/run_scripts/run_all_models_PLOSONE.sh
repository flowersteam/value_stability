#!/bin/bash

experiment_setting="pvq_tolk"
time_req="mid"

#experiment_setting="pvq_fam"
#time_req="mid"

#experiment_setting="don"
#time_req="high"

#experiment_setting="religion"
#time_req="low"

#experiment_setting="bag"
#time_req="high"

## Phi-1 and 2
#################

# Define times based on the given parameter
case "$time_req" in
    low) TIME="00:09:59" ;;
    mid) TIME="00:29:59" ;;
    high) TIME="01:59:59" ;;
esac
sed -i "s/^#SBATCH --time=.*/#SBATCH --time=$TIME/" PLOSONE/run_scripts/run_campaign_seeds.sh

sed -i "s/^#SBATCH --gres=.*/#SBATCH --gres=gpu:1/" PLOSONE/run_scripts/run_campaign_seeds.sh

for model in "phi-1" "phi-2"
do
  sbatch PLOSONE/run_scripts/run_campaign_seeds.sh $model $experiment_setting
done

## Qwen 14B 7B
#################
case "$time_req" in
    low) TIME="00:29:59" ;;
    mid) TIME="00:59:59" ;;
    high) TIME="02:59:59" ;;
esac
sed -i "s/^#SBATCH --time=.*/#SBATCH --time=$TIME/" PLOSONE/run_scripts/run_campaign_seeds.sh

sed -i "s/^#SBATCH --gres=.*/#SBATCH --gres=gpu:1/" PLOSONE/run_scripts/run_campaign_seeds.sh

for model in "Qwen-14B" "Qwen-7B"
do
  sbatch PLOSONE/run_scripts/run_campaign_seeds.sh $model $experiment_setting
done

## llamas and mistrals
######################
case "$time_req" in
    low) TIME="01:09:59" ;;
    mid) TIME="01:09:59" ;;
    high) TIME="01:59:59" ;;
esac

sed -i "s/^#SBATCH --time=.*/#SBATCH --time=$TIME/" PLOSONE/run_scripts/run_campaign_seeds.sh
sed -i "s/^#SBATCH --gres=.*/#SBATCH --gres=gpu:1/" PLOSONE/run_scripts/run_campaign_seeds.sh

models=(
  "llama_2_7b"
  "llama_2_13b"
  "llama_2_7b_chat"
  "llama_2_13b_chat"
  "zephyr-7b-beta"
  "Mistral-7B-v0.1"
  "Mistral-7B-Instruct-v0.1"
  "Mistral-7B-Instruct-v0.2"
)
for model in "${models[@]}"
do
  sbatch PLOSONE/run_scripts/run_campaign_seeds.sh $model $experiment_setting
done

# Mixtrals 4b
################
case "$time_req" in
    low) TIME="00:59:59" ;;
    mid) TIME="02:29:59" ;;
    high) TIME="03:59:59" ;;
esac
sed -i "s/^#SBATCH --time=.*/#SBATCH --time=$TIME/" PLOSONE/run_scripts/run_campaign_seeds.sh

sed -i "s/^#SBATCH --gres=.*/#SBATCH --gres=gpu:1/" PLOSONE/run_scripts/run_campaign_seeds.sh

for model in "Mixtral-8x7B-v0.1-4b" "Mixtral-8x7B-Instruct-v0.1-4b"
do
  sbatch PLOSONE/run_scripts/run_campaign_seeds.sh $model $experiment_setting
done

# Qwen 72B
################
case "$time_req" in
    low) TIME="00:59:59" ;;
    mid) TIME="01:39:59" ;;
    high) TIME="02:59:59" ;;
esac
sed -i "s/^#SBATCH --time=.*/#SBATCH --time=$TIME/" PLOSONE/run_scripts/run_campaign_seeds.sh

sed -i "s/^#SBATCH --gres=.*/#SBATCH --gres=gpu:2/" PLOSONE/run_scripts/run_campaign_seeds.sh

sbatch PLOSONE/run_scripts/run_campaign_seeds.sh "Qwen-72B" $experiment_setting


# LLaMa-2 70b
################
case "$time_req" in
    low) TIME="01:29:59" ;;
    mid) TIME="02:29:59" ;;
    high) TIME="02:59:59" ;;
esac
sed -i "s/^#SBATCH --time=.*/#SBATCH --time=$TIME/" PLOSONE/run_scripts/run_campaign_seeds.sh

sed -i "s/^#SBATCH --gres=.*/#SBATCH --gres=gpu:2/" PLOSONE/run_scripts/run_campaign_seeds.sh

for model in "llama_2_70b" "llama_2_70b_chat"
do
  sbatch PLOSONE/run_scripts/run_campaign_seeds.sh $model $experiment_setting
done


# Mixtrals
################
case "$time_req" in
    low) TIME="00:29:59" ;;
    mid) TIME="00:59:59" ;;
    high) TIME="01:29:59" ;;
esac

sed -i "s/^#SBATCH --time=.*/#SBATCH --time=$TIME/" PLOSONE/run_scripts/run_campaign_seeds.sh
sed -i "s/^#SBATCH --gres=.*/#SBATCH --gres=gpu:4/" PLOSONE/run_scripts/run_campaign_seeds.sh

for model in "Mixtral-8x7B-v0.1" "Mixtral-8x7B-Instruct-v0.1"
do
  sbatch PLOSONE/run_scripts/run_campaign_seeds.sh $model $experiment_setting
done