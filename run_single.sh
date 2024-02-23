#!/bin/bash

# 6 themes
themes=(
  "grammar"
  "joke"
  "poem"
  "history"
  "chess"
  "None"
)

#################################
# 1. Select the theme index [0-5]
#################################

theme_i=0
theme="${themes[$theme_i]}"

##################
## 2. Set the seed
##################
seed="1"

######################################
## 3. Set number of simulated messages
######################################
n_msgs=3


permute_options_seed="$seed"_"$theme_i"

#################################################
## 4. Select the llm (uncomment the one you want)
#################################################

#engine="llama_2_7b"
#engine="llama_2_13b"
#engine="llama_2_7b_chat"
#engine="llama_2_13b_chat"
#engine="zephyr-7b-beta"
#engine="Mistral-7B-v0.1"
#engine="Mistral-7B-Instruct-v0.1"
#engine="Mistral-7B-Instruct-v0.2"
#engine="llama_2_70b" # 2 gpu
#engine="llama_2_70b_chat" # 2 gpu
#engine="Mixtral-8x7B-v0.1-4b" # 4h
#engine="Mixtral-8x7B-Instruct-v0.1-4b" # 4h
#engine="Mixtral-8x7B-v0.1"
#engine="Mixtral-8x7B-Instruct-v0.1"
#engine="phi-2"
#engine="phi-1"
#engine="phi-1.5"
#engine="Qwen-72B"
#engine="Qwen-14B"
#engine="Qwen-7B"
#engine="Qwen-72B-Chat"
engine="dummy"




###########################################################
## 5. Questionnaire (uncomment the corresponding two lines)
###########################################################

# PVQ
test_tag="pvq"
data_dir="data_pvq"
experiment_name="pvq_test"

# Tolkien donation
#test_tag="tolkien_donation"
#data_dir="data_tolkien_donation"
#experiment_name="tolkien_donation_test"


#######################################################
## 6. Simulated population (uncomment the one you want)
#######################################################
population_type="tolkien_characters"
#population_type="famous_people"

#######################################################
## 7. Set experiment name
#######################################################
exp_name="test_experiment"


echo "Your experiment:"
echo "1. Theme:"$theme
echo "2. Seed:"$seed
echo "3. N messages:"$n_msgs
echo "4. LLM:"$engine
echo "5. Questionnaire:"$test_tag
echo "6. Population:"$population_type
echo "7. Experiment name:"$exp_name


### Executing the evaluation

SUBDIR=$exp_name"_"$test_tag"_"$population_type"_seeds/"$engine"/"$seed"_seed/results_sim_conv_"$population_type"_"$engine"_msgs_"$n_msgs
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