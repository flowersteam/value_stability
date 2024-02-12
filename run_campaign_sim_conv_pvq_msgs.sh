#!/bin/bash
#SBATCH -A imi@a100
#SBATCH -C a100
#SBATCH --time=03:59:59
#SBATCH --gres=gpu:4
#SBATCH --array=0-124 # n_msg x n_seeds x n_themes x-> 5x5x5
#SBATCH -o slurm_logs/sb_log_%A_%a.out
#SBATCH -e slurm_logs/sb_log_%A_%a.err
##SBATCH --qos=qos_gpu-dev

#####################################################
### Simulated Conversations
#####################################################

# extract theme n_msgs and seed
n_msgs_list=(9 7 5 3 1) # 5
n_msgs_len=${#n_msgs_list[@]}


seed_list=(1 3 5 7 9)
seed_list_len=${#seed_list[@]}

# 5 themes
themes=(
  "grammar"
  "joke"
  "poem"
  "history"
  "chess"
)
themes_len=${#themes[@]}

echo "ID:"$SLURM_ARRAY_TASK_ID

theme_x_seeds=$(( $themes_len * $seed_list_len ))

msgs_i=$(( SLURM_ARRAY_TASK_ID / $theme_x_seeds ))
theme_i=$(( (( SLURM_ARRAY_TASK_ID % $theme_x_seeds )) / $n_msgs_len ))
seed_i=$(( (( SLURM_ARRAY_TASK_ID  % $theme_x_seeds)) % $themes_len ))

theme="${themes[$theme_i]}"
n_msgs="${n_msgs_list[$msgs_i]}"
seed="${seed_list[$seed_i]}"

echo "Msg_i:"$n_msgs
echo "Theme_i:"$theme
echo "Seed_i:"$seed

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
  "dummy"
#  "gpt-3.5-turbo-0301"
#  "gpt-3.5-turbo-0613"
#  "gpt-3.5-turbo-1106"
#  "gpt-3.5-turbo-instruct-0914"
)

# Select engine based on provided index
engine="${all_engines[$1]}"


## PVQ
test_tag="pvq"
experiment_name="pvq_test"
data_dir="data_pvq"
population_type="tolkien_characters"
#population_type="famous_people"



echo "Evaluation:$engine:$theme:$permute_options_seed:$n_msgs"

SUBDIR="sim_conv_"$test_tag"_"$population_type"_msgs/"$engine"/"$n_msgs"_msgs/"$seed"_seed/results_sim_conv_"$population_type"_"$engine

SAVE_DIR="results/"$SUBDIR
LOG_DIR="logs/"$SUBDIR

mkdir -p $SAVE_DIR
mkdir -p $LOG_DIR

source $HOME/.bashrc

#PY='/gpfsscratch/rech/imi/utu57ed/miniconda3/envs/llm_persp/bin/python'
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