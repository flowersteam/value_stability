#!/bin/bash

#####################################################
### Formats
#####################################################
#
##ENGINE="gpt-4-0314"
#
##ENGINE="gpt-3.5-turbo-0301"
##ENGINE="gpt-3.5-turbo-0613"
##ENGINE="gpt-3.5-turbo-1106"
##ENGINE="gpt-3.5-turbo-instruct-0914"
#
##ENGINE="llama_2_70b_chat"
##ENGINE="llama_2_13b_chat"
##ENGINE="llama_2_70b"
#
##ENGINE="zephyr-7b-beta"
#
#
#population_types=(
##  "user_personas"
##  "llm_personas"
##  "famous_people"
##  "lotr_characters"
#  "tolkien_characters"
#)
#
#format_options=(
##  "chat"
##  "code_cpp"
##  "code_py"
#  "conf_toml"
#  "latex"
#)
#
#for population_type in "${population_types[@]}"; do
#for format in "${format_options[@]}"; do
#
#SAVE_DIR="results_stability/results_pvq_test_format_"$population_type"_"$ENGINE
#mkdir -p $SAVE_DIR
#
#
#
#python -u evaluate.py \
#--simulated-population-type $population_type \
#--save_dir $SAVE_DIR \
#--engine "$ENGINE" \
#--query-in-reply \
#--system-message \
#--data_dir data_pvq \
#--experiment_name pvq_test \
#--ntrain 0 \
#--format $format \
#--no-profile \
#--verbose  2>&1 | tee -a $SAVE_DIR/log_$format.txt
#
#done
#done


#####################################################
### Themes (Simulated Conversations)
#####################################################

#ENGINE="gpt-4-0314"

#ENGINE="gpt-3.5-turbo-0301"
#ENGINE="gpt-3.5-turbo-0613"
#ENGINE="gpt-2.5-turbo-1106"
#ENGINE="gpt-3.5-turbo-instruct-0914"

ENGINE="llama_2_70b_chat"

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

SAVE_DIR="results_stability/results_pvq_test_sim_conv_"$population_type"_"$ENGINE
mkdir -p $SAVE_DIR


# conversation
python -u evaluate.py \
--simulated-population-type $population_type \
--save_dir $SAVE_DIR \
--query-in-reply \
--engine "$ENGINE" \
--system-message \
--data_dir data_pvq \
--experiment_name pvq_test \
--ntrain 0 \
--format chat \
--simulate-conversation-theme $theme \
--no-profile \
--verbose  2>&1 | tee -a $SAVE_DIR/log_$theme.txt

done
done

######################################################
#### Weather
######################################################
#
##ENGINE="gpt-4-0314"
#
##ENGINE="gpt-3.5-turbo-0301"
##ENGINE="gpt-3.5-turbo-0613"
##ENGINE="gpt-3.5-turbo-1106"
##ENGINE="gpt-3.5-turbo-instruct-0914"
#
#ENGINE="llama_2_70b_chat"
##ENGINE="zephyr-7b-beta"
#
#
#population_types=(
##  "user_personas"
##  "llm_personas"
##  "famous_people"
##  "lotr_characters"
#  "tolkien_characters"
#)
#
#weathers=(
#  "rain"
#  "sun"
#  "snow"
#  "thunderstorm"
#  "sandstorm"
#  "blizzard"
#)
#
#for weather in "${weathers[@]}"; do
#for population_type in "${population_types[@]}"; do
#
#SAVE_DIR="results_stability/results_pvq_test_weather_conv_"$population_type"_"$ENGINE
#mkdir -p $SAVE_DIR
#
#
## conversation
#python -u evaluate.py \
#--simulated-population-type $population_type \
#--save_dir $SAVE_DIR \
#--engine "$ENGINE" \
#--query-in-reply \
#--system-message \
#--data_dir data_pvq \
#--experiment_name pvq_test \
#--ntrain 0 \
#--format chat \
#--weather $weather \
#--no-profile \
#--verbose  2>&1 | tee -a $SAVE_DIR/log_$theme.txt
#
#done
#done
