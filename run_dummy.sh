#!/bin/bash

##########################################################
# Set the questionnaire and population (using the second command argument)
##########################################################

theme="chess"
n_msgs=3
permute_options_seed="test"
#engine="dummy"
#engine="interactive"

#engine="gpt-3.5-turbo-0125"

#engine="phi-3"
#engine="Qwen1.5-72B-Chat"

#engine="Mistral-7B-Instruct-v0.1"
#engine="Mixtral-8x7B-Instruct-v0.1-4b"

#engine="Mixtral-8x22B-Instruct-v0.1-4b" # 2gpus 3.5h
#engine="Mixtral-8x22B-Instruct-v0.1" # 2gpus 3.5h
#engine="command_r_plus-4b" # 2gpus 5h

engine="llama_3_8b_instruct"  # 1 gpu, 1.5
#engine="llama_3_70b_instruct" # 2 gpu, 1.5

#experiment_setting=religion

#engine="llama_2_7b"
#engine="llama_2_13b"
#engine="llama_2_70b"
#engine="llama_2_7b_chat"
#engine="llama_2_13b_chat"
#engine="llama_2_70b_chat"
#engine="Mistral-7B-v0.1"
#engine="Mistral-7B-Instruct-v0.1"
#engine="Mistral-7B-Instruct-v0.2"
#engine="zephyr-7b-beta"
#engine="Mixtral-8x7B-v0.1-4b"
#engine="Mixtral-8x7B-Instruct-v0.1-4b"
#engine="Mixtral-8x7B-v0.1"
#engine="Mixtral-8x7B-Instruct-v0.1"
#engine="phi-1"
#engine="phi-2"
#engine="Qwen-7B"
#engine="Qwen-14B"
#engine="Qwen-72B"
#engine="gpt-3.5-turbo-1106"
#engine="gpt-3.5-turbo-0125"


experiment_setting=pvq_tolk

# Define the configuration based on the experiment_setting
case "$experiment_setting" in
  pvq_tolk)
    test_tag="pvq"
    experiment_name="pvq_test"
    data_dir="data_pvq"
    population_type="tolkien_characters"
    ;;
  pvq_fam)
    test_tag="pvq"
    experiment_name="pvq_test"
    data_dir="data_pvq"
    population_type="famous_people"
    ;;
  don)
    test_tag="tolkien_donation"
    experiment_name="tolkien_donation_test"
    data_dir="data_tolkien_donation"
    population_type="tolkien_characters"
    ;;
  bag)
    test_tag="tolkien_bag"
    experiment_name="tolkien_bag_test"
    data_dir="data_tolkien_bag"
    population_type="tolkien_characters"
    ;;
  religion)
    test_tag="religion"
    experiment_name="religion_test"
    data_dir="data_religion"
    population_type="famous_people"
    ;;
  *)
    echo "Invalid experiment_setting. Please use one of the following: pvq_tolk, pvq_fam, don, bag, religion."
    exit 1
    ;;
esac

# Print the selected configuration
echo "test_tag=$test_tag"
echo "experiment_name=$experiment_name"
echo "data_dir=$data_dir"
echo "population_type=$population_type"
echo "engine=$engine"
#####################################################


SUBDIR="test/"$engine"/"$seed"_seed/results_sim_conv_"$population_type"_"$engine"_msgs_"$n_msgs
SAVE_DIR="test_results/"$SUBDIR
LOG_DIR="test_logs/"$SUBDIR

#mkdir -p $SAVE_DIR
mkdir -p $LOG_DIR

source $HOME/.bashrc

case "$engine" in
    phi-1|phi-2|Qwen1.5*|llama_3*|command_r_plus*|Mixtral-8x22B*)
        conda activate llm_stability_phi
        ;;
    *)
        conda activate llm_stability
        ;;
esac


python -u evaluate.py \
  --simulated-population-config $population_type \
  --simulated-conversation-theme $theme \
  --interlocutor-knows-persona \
  --simulated-conversation-n-messages $n_msgs \
  --permute-options-seed "$permute_options_seed" \
  --format chat \
  --save_dir $SAVE_DIR \
  --engine "$engine" \
  --data_dir data/$data_dir \
  --experiment_name $experiment_name \
  --pvq-version "pvq_auto" \
  --assert-params \
  --verbose  2>&1 | tee -a $LOG_DIR/log_$permute_options_seed.txt