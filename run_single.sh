#!/bin/bash


# 6 themes

#################################
# 1. Select the theme index [0-5]
#################################

eval_i=0

#themes=( "grammar" "joke" "poem" "history" "chess" )
#theme="${themes[$eval_i]}"
#theme="contexts/superlong_reddit_chunks/chunk_0.jsonl"
#theme="contexts/mixed_v2_reddit_chunks/chunk_{eval_i}.jsonl"
#theme="chess"
#theme="bjj"

if [ $eval_i -eq 0 ]; then
  # no conversation
  chunk_i="no_conv"
  theme="None"
  interlocutors="human"

elif [ $eval_i -eq 6 ]; then
  # classic themes (chess), interlocutors chunk 0
  chunk_i="chess_"$[eval_i - 6]
  theme="chess"

  interlocutors_chunk=0
  interlocutors="./interlocutors/real_world_people/chunk_"$interlocutors_chunk"/interlocutors.json"

elif [ $eval_i -eq 7 ]; then
  # classic themes (grammar), interlocutors chunk 1
  chunk_i="grammar_"$[eval_i - 6]
  theme="grammar"

  interlocutors_chunk=1
  interlocutors="./interlocutors/real_world_people/chunk_"$interlocutors_chunk"/interlocutors.json"

elif [ $eval_i -ge 8 ]; then
  # undefined
  echo "Undefined eval_i:$eval_i"
  exit

else
  # 1 - 5 -> chunks: 0 - 4
  # reddit chunks
  chunk_i=$[eval_i - 1]
  theme="contexts/leaderboard_reddit_chunks/chunk_"$chunk_i".jsonl"
#  theme="contexts/mixed_v2_reddit_chunks/chunk_"$chunk_i".jsonl"

  interlocutors="human"
fi


##################
## 2. Set the seed
##################
seed="1"

######################################
## 3. Set number of simulated messages
######################################
n_msgs=3

#n_msgs=31
#n_msgs=7


permute_options_seed="$seed"_"$chunk_i"

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
#engine="Qwen-72B"
#engine="Qwen-14B"
#engine="Qwen-7B"
#engine="Qwen-72B-Chat"


#model_config="./models/leaderboard_configs/dummy.json"

#model_config="./models/leaderboard_configs/phi-3.json"
#model_config="./models/leaderboard_configs/llama_3_8b_instruct.json"
#model_config="./models/leaderboard_configs/llama_3_8b_instruct_vllm.json"
#model_config="./models/leaderboard_configs/llama_3_70b_instruct.json"

#engine="llama_3_8b_instruct_vllm"
#engine="llama_3_70b_instruct"
#engine="phi-3_vllm"

#engine="phi-3-mini-128k-instruct"
#engine="gpt-3.5-turbo-0125"
#engine="gpt-4o-0513"

#engine="Mixtral-8x7B-Instruct-v0.1"
#engine="Mistral-7B-Instruct-v0.3"
#engine="Mistral-7B-Instruct-v0.1"

#engine="phi-2"
#engine="phi-3-medium-128k-instruct"
#engine="command_r_plus"
#engine="Qwen2-72B-Instruct"
#engine="Qwen2-7B-Instruct"
#engine="gpt-3.5-turbo-0125"
#engine="dummy"
engine="llama_3_8b_instruct"

model_config="./models/leaderboard_configs/$engine.json"

###########################################################
## 5. Questionnaire (uncomment the corresponding two lines)
###########################################################

# PVQ
#test_tag="pvq"
#data_dir="data_pvq"
#experiment_name="pvq_test"

test_tag="svs"
data_dir="data_svs"
experiment_name="svs_test"

# Tolkien donation
#test_tag="tolkien_donation"
#data_dir="data_tolkien_donation"
#experiment_name="tolkien_donation_test"


#######################################################
## 6. Simulated population (uncomment the one you want)
#######################################################
#population_config="tolkien_characters"
#population_config="famous_people"
population_config="./personas/real_world_people/personas.json"

#######################################################
## 7. Interlocutors
#######################################################
#interlocutors="./interlocutors/real_world_people/chunk_0/interlocutors.json"
#interlocutors="human"

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

SUBDIR="single"
SAVE_DIR="results/"$SUBDIR
LOG_DIR="logs/"$SUBDIR

mkdir -p $SAVE_DIR
mkdir -p $LOG_DIR

#source $HOME/.bashrc

#if [[ "$model_config" == "*vllm*" ]]; then
#    conda activate llm_stability_vllm
#elif [[ "$model_config" == "*/phi-1.json" || "$engine" == "*/phi-2.json" ]]; then
#    conda activate llm_stability_phi
#else
#    conda activate llm_stability
#fi

#source $HOME/.bashrc
#
#case "$engine" in
#    *vllm)
#      module load python/3.9.12
#      conda activate llm_stability_vllm
#      ;;
#    phi-1|phi-2|Qwen1.5*|llama_3*|command_r_plus*|Mixtral-8x22B*)
#      module load python/3.10.4
#      conda activate llm_stability_437
#      ;PLOSONE;
#    *)
#      module load python/3.10.4
#      conda activate llm_stability
#      ;;
#esac

#module load python/3.10.4
#conda activate llm_stability_441

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
