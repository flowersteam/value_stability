#!/bin/bash

#####################################################
### Themes (Simulated Conversations)
#####################################################

#ENGINE="gpt-4-0314"

#ENGINE="gpt-3.5-turbo-0301"
ENGINE="gpt-3.5-turbo-0613"
#ENGINE="gpt-2.5-turbo-1106"

#ENGINE="gpt-3.5-turbo-instruct-0914"

#ENGINE="llama_2_13b_chat"
#ENGINE="llama_2_70b"

#ENGINE="zephyr-7b-beta"


population_types=(
#  "user_personas"
#  "llm_personas"
#  "famous_people"
#  "lotr_characters"
  "tolkien_characters"
)

themes=(
  "grammar"
  "joke"
  "poem"
  "history"
  "chess"
)

# more like priming
#  "religion"
#  "tax"
#  "vacation"

for theme in "${themes[@]}"; do
for population_type in "${population_types[@]}"; do

SAVE_DIR="results_stability/results_ultimatum_test_sim_conv_"$population_type"_"$ENGINE
mkdir -p $SAVE_DIR

#--query-in-reply \

# conversation
python -u evaluate.py \
--simulated-population-type $population_type \
--save_dir $SAVE_DIR \
--engine "$ENGINE" \
--system-message \
--data_dir data_ultimatum \
--experiment_name ultimatum_test \
--ntrain 0 \
--format chat \
--simulate-conversation-theme $theme \
--no-profile \
--verbose  2>&1 | tee -a $SAVE_DIR/log_$theme.txt

done
done