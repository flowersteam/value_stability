##!/bin/bash


###########################################
# Chat/Code
###########################################

#ENGINE="gpt-4-0314"
#ENGINE="gpt-3.5-turbo-0301"
ENGINE="gpt-3.5-turbo-0613"
##ENGINE="gpt-3.5-turbo-instruct-0914"

#ENGINE="up_llama2_70b_instruct_v2"
#ENGINE="up_llama_60b_instruct"

#ENGINE="zephyr-7b-beta"

#ENGINE="openassistant_rlhf2_llama30b"
##ENGINE="stablevicuna"
##ENGINE="stablelm"
##ENGINE="dummy"
#
##PERMUTATIONS=1
PERMUTATIONS=50

####### Formats
format_options=(
#  "chat"
#  "code_py"
  "code_cpp"
#  "conf_toml"
#  "latex"
)

for format in "${format_options[@]}"; do

SAVE_DIR="results_plosone/results_pvq_test_gen_qa_"$population_type"_format_"$ENGINE"_perm_"$PERMUTATIONS"_format"
mkdir -p $SAVE_DIR

python -u evaluate.py \
--simulated-population-type permutations \
--permutations $PERMUTATIONS \
--save_dir $SAVE_DIR \
--engine "$ENGINE" \
--generative-qa \
--data_dir data_pvq \
--experiment_name pvq_test \
--ntrain 0 \
--format $format \
--no-profile \
--verbose  2>&1 | tee -a $SAVE_DIR/log.txt

#--estimate-gpt-tokens \

done

########### Themes (Sim Conv)
#themes=(
##  "religion"
##  "tax"
##  "vacation"
#  "grammar"
#  "poem"
#  "joke"
#  "history"
#  "chess"
#)
#
#for theme in "${themes[@]}"; do
#
#SAVE_DIR="results_iclr/results_pvq_test_sim_conv_"$ENGINE"_perm_"$PERMUTATIONS"_theme"
#mkdir -p $SAVE_DIR
## conversation
#python -u evaluate.py \
#--permutations 50 \
#--save_dir $SAVE_DIR \
#--engine $ENGINE \
#--data_dir data_pvq \
#--experiment_name test_pvq \
#--ntrain 0 \
#--format chat \
#--no-profile \
#--simulate-conversation-theme $theme \
#--system-message \
#--verbose  2>&1 | tee -a $SAVE_DIR/log.txt
#
#done

########## Themes - with set values
#themes=(
#  "poem"
#  "joke"
#  "history"
#  "grammar"
#  "chess"
#)
#
#vals="Universalism,Benevolence"
#
#for theme in "${themes[@]}"; do
#
#SAVE_DIR="results_iclr/results_pvq_test_sim_conv_2nd_prs_explicit_values_"$vals"_"$ENGINE"_perm_"$PERMUTATIONS"_theme"
#mkdir -p $SAVE_DIR
#
## conversation
#python -u evaluate.py \
#--permutations 50 \
#--save_dir $SAVE_DIR \
#--engine $ENGINE \
#--data_dir data_pvq \
#--experiment_name test_pvq \
#--separator \
#--ntrain 0 \
#--format chat \
#--simulate-conversation-theme $theme \
#--system-message \
#--direct-perspective \
#--profile "Primary values:$vals" \
#--natural-language-profile \
#--natural-language-profile-detail "no" \
#--perspective-amount "extreme" \
#--add-high-level-categories \
#--verbose  2>&1 | tee -a $SAVE_DIR/log.txt
#
#done

############# WIKIPEDIA
#music_genre_list=(
#  "hip-hop"
#  "jazz"
#  "classical"
#  "heavy metal"
#  "reggae"
#  "gospel"
#)
## in this experiments person doesn't have any influence as part1 of the prompt is from wikipedia
#message="User"
#person="3rd"
#PERMUTATIONS=50
##ENGINE="gpt-4-0314"
#ENGINE="gpt-3.5-turbo-0301"
##ENGINE="gpt-4-0613"
##ENGINE="gpt-3.5-turbo-0613"
##ENGINE="dummy"
##ENGINE="openassistant_rlhf2_llama30b"
##ENGINE="stablevicuna"
#
#for music_genre in "${music_genre_list[@]}"; do
#
#SAVE_DIR="results_iclr/results_AI_wiki_context_v2_no_separator_music_expert_pvq_test_"$ENGINE"_perm_"$PERMUTATIONS"_"$message"_msg_"$person"_prs"
#
#python evaluate.py \
#--permutations $PERMUTATIONS \
#--save_dir $SAVE_DIR \
#--engine "$ENGINE" \
#--data_dir data_pvq \
#--experiment_name test_pvq \
#--ntrain 0 \
#--verbose \
#$(if [ "$message" == "System" ]; then echo "--system-message"; fi) \
#$(if [ "$person" == "2nd" ]; then echo "--direct-perspective"; fi) \
#--music-expert-genre  "$music_genre" \
#--wiki-context

#--estimate-gpt-tokens \

#done
