#!/bin/bash
#SBATCH -A imi@a100
#SBATCH -C a100
#SBATCH --time=03:59:59
#SBATCH --gres=gpu:1
#SBATCH --array=0-29 # themes x n_seeds -> 6x5
#SBATCH -o slurm_logs/sb_log_%A_%a.out
#SBATCH -e slurm_logs/sb_log_%A_%a.err
##SBATCH --qos=qos_gpu-dev


##########################################################
# Set the questionnaire and population (uncomment he corresponding 4 lines)
##########################################################

## PVQ - tolkien characters
test_tag="pvq"
experiment_name="pvq_test"
data_dir="data_pvq"
population_type="tolkien_characters"

### PVQ - real world persona
#test_tag="pvq"
#experiment_name="pvq_test"
#data_dir="data_pvq"
#population_type="famous_people"

## Donation - tolkien characters
#test_tag="tolkien_donation"
#experiment_name="tolkien_donation_test"
#data_dir="data_tolkien_donation"
#population_type="tolkien_characters"

#####################################################



# extract theme and n_msgs
seed_list=(1 3 5 7 9)

seed_list_len=${#seed_list[@]}

# 6 themes
themes=(
  "grammar"
  "joke"
  "poem"
  "history"
  "chess"
  "None"
)
themes_len=${#themes[@]}

echo "ID:"$SLURM_ARRAY_TASK_ID

theme_i=$(( SLURM_ARRAY_TASK_ID / $seed_list_len ))
seed_i=$(( SLURM_ARRAY_TASK_ID % $seed_list_len ))


theme="${themes[$theme_i]}"
seed="${seed_list[$seed_i]}"

echo "Theme_i:"$theme_i
echo "Theme:"$theme

echo "Seed_i:"$seed_i
echo "Seed:"$seed


n_msgs=3

permute_options_seed="$seed"_"$theme_i"

all_engines=(
  "llama_2_7b"
  "llama_2_13b"
  "llama_2_7b_chat"
  "llama_2_13b_chat"
  "zephyr-7b-beta"
  "Mistral-7B-v0.1"
  "Mistral-7B-Instruct-v0.1"
  "Mistral-7B-Instruct-v0.2"
  "llama_2_70b" # 2 gpu
  "llama_2_70b_chat" # 2 gpu
  "Mixtral-8x7B-v0.1-4b" # 4h
  "Mixtral-8x7B-Instruct-v0.1-4b" # 4h
  "Mixtral-8x7B-v0.1"
  "Mixtral-8x7B-Instruct-v0.1"
  "phi-2"
  "phi-1"
  "phi-1.5"
  "Qwen-72B"
  "Qwen-14B"
  "Qwen-7B"
  "Qwen-72B-Chat"
  "dummy"
)

# Select engine based on provided index
engine="${all_engines[$1]}"


echo "Evaluation:$engine:$theme:$permute_options_seed:$n_msgs"

SUBDIR="sim_conv_"$test_tag"_"$population_type"_seeds/"$engine"/"$seed"_seed/results_sim_conv_"$population_type"_"$engine"_msgs_"$n_msgs
SAVE_DIR="results/"$SUBDIR
LOG_DIR="logs/"$SUBDIR

mkdir -p $SAVE_DIR
mkdir -p $LOG_DIR

source $HOME/.bashrc

conda activate llm_persp


if [[ $engine == *"Mistral"* ]] || [[ $engine == *"Mixtral"* ]]; then

  echo "Mistral or Mixtral: $engine"

  if [[ $engine == *"Instruct"* ]] ; then
    # INSTRUCT MODELS

    # mistral, mixtral -> no sys; query
    python -u evaluate.py \
      --simulated-population-type $population_type \
      --simulate-conversation-theme $theme \
      --simulated-human-knows-persona \
      --simulated-conversation-n-messages $n_msgs \
      --permute-options \
      --permute-options-seed "$permute_options_seed" \
      --format chat \
      --save_dir $SAVE_DIR \
      --engine "$engine" \
      --query-in-reply \
      --data_dir data/$data_dir \
      --experiment_name $experiment_name \
      --pvq-version "pvq_auto" \
      --no-profile \
      --assert-params \
      --verbose  2>&1 | tee -a $LOG_DIR/log_$permute_options_seed.txt

  else
    # BASE MODELS

    # mistral, mixtral -> no sys; query
    python -u evaluate.py \
      --simulated-population-type $population_type \
      --simulate-conversation-theme $theme \
      --simulated-human-knows-persona \
      --simulated-conversation-n-messages $n_msgs \
      --permute-options \
      --permute-options-seed "$permute_options_seed" \
      --format chat \
      --save_dir $SAVE_DIR \
      --engine "$engine" \
      --query-in-reply \
      --system-message \
      --base-model-template \
      --data_dir data/$data_dir \
      --experiment_name $experiment_name \
      --pvq-version "pvq_auto" \
      --no-profile \
      --assert-params \
      --verbose  2>&1 | tee -a $LOG_DIR/log_$permute_options_seed.txt
  fi

elif [[ $engine == *"phi"* ]] || [[ $engine == "Qwen-"*"B" ]]; then

    # all phi models are BASE and qwen base
    python -u evaluate.py \
      --simulated-population-type $population_type \
      --simulate-conversation-theme $theme \
      --simulated-human-knows-persona \
      --simulated-conversation-n-messages $n_msgs \
      --permute-options \
      --permute-options-seed "$permute_options_seed" \
      --format chat \
      --save_dir $SAVE_DIR \
      --engine "$engine" \
      --query-in-reply \
      --system-message \
      --base-model-template \
      --data_dir data/$data_dir \
      --experiment_name $experiment_name \
      --pvq-version "pvq_auto" \
      --no-profile \
      --assert-params \
      --verbose  2>&1 | tee -a $LOG_DIR/log_$permute_options_seed.txt

elif [[ $engine == *"zephyr"* ]] || [[ $engine == *"llama_2"* ]] || [[ $engine == "dummy" ]]; then

  echo "Zephyr or LLaMa: $engine"

  if [[ $engine == *"llama_2"* ]] && [[ $engine != *"chat"* ]]; then
    # BASE MODELS

    # llama_base_model
    python -u evaluate.py \
      --simulated-population-type $population_type \
      --simulate-conversation-theme $theme \
      --simulated-human-knows-persona \
      --simulated-conversation-n-messages $n_msgs \
      --permute-options \
      --permute-options-seed "$permute_options_seed" \
      --format chat \
      --save_dir $SAVE_DIR \
      --engine "$engine" \
      --query-in-reply \
      --system-message \
      --base-model-template \
      --data_dir data/$data_dir \
      --experiment_name $experiment_name \
      --pvq-version "pvq_auto" \
      --no-profile \
      --assert-params \
      --verbose  2>&1 | tee -a $LOG_DIR/log_$permute_options_seed.txt

  else

    # INSTUCT, DPO models
    # zephyr, llama -> sys ; query
    python -u evaluate.py \
      --simulated-population-type $population_type \
      --simulate-conversation-theme $theme \
      --simulated-human-knows-persona \
      --simulated-conversation-n-messages $n_msgs \
      --permute-options \
      --permute-options-seed "$permute_options_seed" \
      --format chat \
      --save_dir $SAVE_DIR \
      --engine "$engine" \
      --query-in-reply \
      --system-message \
      --data_dir data/$data_dir \
      --experiment_name $experiment_name \
      --pvq-version "pvq_auto" \
      --no-profile \
      --assert-params \
      --verbose  2>&1 | tee -a $LOG_DIR/log_$permute_options_seed.txt

  fi


elif [[ $engine == *"gpt"* ]] ; then

  echo "GPTs: $engine"

  # gpts -> sys ; no query
  python -u evaluate.py \
    --simulated-population-type $population_type \
    --simulate-conversation-theme $theme \
    --simulated-human-knows-persona \
    --simulated-conversation-n-messages $n_msgs \
    --permute-options-seed "$permute_options_seed" \
    --permute-options \
    --format chat \
    --save_dir $SAVE_DIR \
    --engine "$engine" \
    --system-message \
    --data_dir data/$data_dir \
    --experiment_name $experiment_name \
    --pvq-version "pvq_auto" \
    --no-profile \
    --assert-params \
    --verbose  2>&1 | tee -a $LOG_DIR/log_$permute_options_seed.txt


else
  echo "Undefined engine: $engine"
fi