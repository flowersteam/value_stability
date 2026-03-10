#!/bin/bash

#################################
# 1. Select the theme
#################################

#theme="None"
#theme="grammar"
#theme="chess"
#theme="contexts/leaderboard_reddit_chunks/chunk_0.jsonl"
theme="contexts/leaderboard_reddit_chunks/chunk_3.jsonl"
#theme="contexts/leaderboard_reddit_chunks/chunk_4.jsonl"


##################
## 2. Set the seed
##################
seed="1"

######################################
## 3. Set number of simulated messages
######################################
n_msgs=3


permute_options_seed="test_seed_$seed"

#################################################
## 4. Select the llm (uncomment the one you want)
#################################################

#engine="gpt-3.5-turbo-0125"
#engine="dummy"
#engine="llama_3_8b_instruct"
engine="gemma-2-9b-it"

model_config="./models/leaderboard_configs/$engine.json"

###########################################################
## 5. Questionnaire (uncomment the corresponding two lines)
###########################################################

# PVQ
test_tag="pvq"
data_dir="data_pvq"
experiment_name="pvq_test"

# SVS
#test_tag="svs"
#data_dir="data_svs"
#experiment_name="svs_test"



#######################################################
## 6. Simulated population (uncomment the one you want)
#######################################################
population_config="./personas/real_world_people/personas.json"

#######################################################
## 7. Interlocutors
#######################################################
#interlocutors="./interlocutors/real_world_people/chunk_0/interlocutors.json"
interlocutors="human"

#######################################################
## 8. Set experiment name
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
echo "7. Interlocutors:"$interlocutors
echo "8. Experiment name:"$exp_name


### Executing the evaluation

SUBDIR="test_single"
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
  --interlocutors $interlocutors \
  --save_dir $SAVE_DIR \
  --pvq-version "pvq_auto" \
  --assert-params \
  --overwrite \
  --verbose # 2>&1 | tee -a $LOG_DIR/log_$permute_options_seed.txt

#  --interlocutor-knows-persona \
