#!/bin/bash


# 6 themes

#################################
# 1. Select the theme
#################################
#themes=( "grammar" "joke" "poem" "history" "chess" )
theme="grammar"

##################
## 2. Set the seed
##################
seed="1"

######################################
## 3. Set number of simulated messages
######################################
n_msgs=3


permute_options_seed="test_permutation"

#################################################
## 4. Select the llm (uncomment the one you want)
#################################################

engine="Mistral-7B-Instruct-v0.2"
model_config="./models/configs/$engine.json"

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
#population_config="./personas/famous_people/personas.json"
population_config="./personas/tolkien_characters/personas.json"


#######################################################
## 7. Set experiment name
#######################################################
exp_name="test_experiment"


echo "Your experiment:"
echo "1. Theme:"$theme
echo "2. Seed:"$seed
echo "2. Permutation Seed:"$permute_options_seed
echo "3. N messages:"$n_msgs
echo "4. LLM:"$model_config
echo "5. Questionnaire:"$test_tag
echo "6. Population:"$population_config
echo "7. Experiment name:"$exp_name


### Executing the evaluation

SUBDIR="single"
SAVE_DIR="results/"$SUBDIR
LOG_DIR="logs/"$SUBDIR

mkdir -p $SAVE_DIR
mkdir -p $LOG_DIR

conda activate llm_stability

python -u evaluate.py \
  --model-config-path $model_config \
  --experiment_name $experiment_name \
  --data_dir data/$data_dir \
  --simulated-population-config $population_config \
  --simulated-conversation-theme $theme \
  --simulated-conversation-n-messages $n_msgs \
  --permute-options-seed "$permute_options_seed" \
  --interlocutors "human" \
  --save_dir $SAVE_DIR \
  --pvq-version "pvq_auto" \
  --assert-params \
  --overwrite \
  --verbose