##!/bin/bash


###########################################
# 1. Implied values (directly / indirectly)
###########################################

# LoTR characters
##################

# 2ps (direct pers) ; System message

##ENGINE="gpt-4-0314"
##ENGINE="gpt-3.5-turbo-0301"
##ENGINE="openassistant_rlhf2_llama30b"
##ENGINE="stablevicuna"
##ENGINE="stablelm"
#ENGINE="dummy"

#lotr_characters=(
#  "Gandalf"
#  "Frodo"
#  "Sauron"
#  "Aragorn"
#  "Pippin"
#)
#
##PERMUTATIONS=50
#PERMUTATIONS=1
#message="System"
#person="2nd"
#
#for character in "${lotr_characters[@]}"; do
#
#SAVE_DIR="results_neurips/results_lotr_pvq_test_"$ENGINE"_perm_"$PERMUTATIONS"_"$message"_msg_"$person"_prs"
#mkdir -p $SAVE_DIR
#
#python -u evaluate.py \
#--permutations $PERMUTATIONS \
#--save_dir $SAVE_DIR \
#--engine "$ENGINE" \
#--data_dir data_pvq \
#--experiment_name pvq_test \
#--separator \
#--ntrain 0 \
#$(if [ "$message" == "System" ]; then echo "--system-message"; fi) \
#$(if [ "$person" == "2nd" ]; then echo "--direct-perspective"; fi) \
#--estimate-gpt-tokens \
#--add-high-level-categories \
#--lotr-character "$character" \
#--verbose  2>&1 | tee -a $SAVE_DIR/log.txt
#
#done

#############################
# 2. Non-implied values
#############################

#### Music AI Experts
#####################
#
#music_genre_list=(
#  "hip-hop"
#  "jazz"
#  "classical"
#  "heavy metal"
#  "reggae"
#  "gospel"
#)
#
#message="System"
#person="2nd"
#
#PERMUTATIONS=50
#
##ENGINE="gpt-4-0314"
#ENGINE="gpt-3.5-turbo-0301"
#
#for music_genre in "${music_genre_list[@]}"; do
#
#SAVE_DIR="results_neurips/results_AI_music_expert_pvq_test_"$ENGINE"_perm_"$PERMUTATIONS"_"$message"_msg_"$person"_prs"
#
#python evaluate.py \
#--permutations $PERMUTATIONS \
#--save_dir $SAVE_DIR \
#--engine "$ENGINE" \
#--data_dir data_pvq \
#--experiment_name pvq_test \
#--separator \
#--ntrain 0 \
#--verbose \
#--estimate-gpt-tokens \
#$(if [ "$message" == "System" ]; then echo "--system-message"; fi) \
#$(if [ "$person" == "2nd" ]; then echo "--direct-perspective"; fi) \
#--music-expert-genre  "$music_genre"
#
#done

####################################
### APP: different contexts
####################################
#text_type_list=(
#  "wiki"
#  "code"
#  "recipe"
#  "poem"
#)
## in this expeirmetns person doesn't have any influence as part1 of the prompt is from wikipedia
##message="User"
#message="User"
#person="3rd"
#PERMUTATIONS=1
##ENGINE="gpt-4-0314"
#ENGINE="gpt-4-0613"
##ENGINE="gpt-3.5-turbo-0301"
##ENGINE="gpt-3.5-turbo-0613"
##ENGINE="dummy"
#
#for text_type in "${text_type_list[@]}"; do
#
#SAVE_DIR="results_neurips/results_text_types_no_separator_pvq_test_"$ENGINE"_perm_"$PERMUTATIONS"_"$message"_msg_"$person"_prs"
#
#python evaluate.py \
#--permutations $PERMUTATIONS \
#--save_dir $SAVE_DIR \
#--engine "$ENGINE" \
#--data_dir data_pvq \
#--experiment_name test_pvq \
#--ntrain 0 \
#--verbose \
#--estimate-gpt-tokens \
#$(if [ "$message" == "System" ]; then echo "--system-message"; fi) \
#$(if [ "$person" == "2nd" ]; then echo "--direct-perspective"; fi) \
#--context-type  "$text_type" \
#--different-contexts
#
#done

####################################
### APP: Music genres wikipedia
####################################
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
##ENGINE="gpt-3.5-turbo-0301"
##ENGINE="gpt-4-0613"
#ENGINE="gpt-3.5-turbo-0613"
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
#
##--estimate-gpt-tokens \
#
#done

## Hobbies
##################
#hobbies_list=(
#  "singing in a church choir"
#  "gambling"
#  "playing chess"
#  "volunteering at a homeless shelter"
#  "car racing"
#)
#
#message="System"
#person="2nd"
#PERMUTATIONS=1
#ENGINE="gpt-4-0314"
#
#for hobby in "${hobbies_list[@]}"; do
#
#SAVE_DIR="results_neurips/results_hobbies_pvq_test_"$ENGINE"_perm_"$PERMUTATIONS"_"$message"_msg_"$person"_prs"
#
#python evaluate.py \
#--permutations $PERMUTATIONS \
#--save_dir $SAVE_DIR \
#--engine "$ENGINE" \
#--data_dir data_pvq \
#--experiment_name pvq_test \
#--separator \
#--ntrain 0 \
#--verbose \
#--estimate-gpt-tokens \
#$(if [ "$message" == "System" ]; then echo "--system-message"; fi) \
#$(if [ "$person" == "2nd" ]; then echo "--direct-perspective"; fi) \
#--hobby  "$hobby"
#
#done

####################################
### 3. System/User 2ne/3rd person
####################################

#### PVQ
########################

pvq_values_list=(
  "Hedonism,Stimulation,Self-Direction"
  "Universalism,Benevolence"
  "Conformity,Tradition,Security"
  "Power,Achievement"
)

message_options=(
#  "System"
  "User"
)

person_options=(
  "2nd"
#  "3rd"
)

# RedPajama INCITE Chat and Instruct variants,
# MPT Chat and Instruct variants
# Alpaca,
# Vicuna,
# Koala fine-tune of the LLaMA models
# Dolly 12B fine-tune of Pythia
# Chat fine-tune of GPT NeoX  20B
# GPT3 through the OpenAI API

#ENGINE="gpt-4-0314"
#ENGINE="gpt-3.5-turbo-0301"
#ENGINE="gpt-3.5-turbo-0613"
ENGINE="gpt-3.5-turbo-instruct-0914"

#ENGINE="openassistant_rlhf2_llama30b"
#ENGINE="stablevicuna"
#ENGINE="stablelm"

#ENGINE="rp_incite_7b_instruct"
#ENGINE="rp_incite_7b_chat"

#ENGINE="up_llama_60b_instruct"
#ENGINE="up_llama2_70b_instruct_v2"

#ENGINE="curie"
#ENGINE="babbage"
#ENGINE="ada"
#ENGINE="text-davinci-003"
#ENGINE="dummy"

PERMUTATIONS=50

for message in "${message_options[@]}"; do
for person in "${person_options[@]}"; do
for vals in "${pvq_values_list[@]}"; do

SAVE_DIR="results_icml/results_nat_lang_prof_pvq_test_"$ENGINE"_perm_"$PERMUTATIONS"_"$message"_msg_"$person"_prs"

mkdir -p $SAVE_DIR

python -u evaluate.py \
--permutations $PERMUTATIONS \
--save_dir $SAVE_DIR \
--engine "$ENGINE" \
--data_dir data_pvq \
--experiment_name pvq_test \
--separator \
--ntrain 0 \
$(if [ "$message" == "System" ]; then echo "--system-message"; fi) \
$(if [ "$person" == "2nd" ]; then echo "--direct-perspective"; fi) \
--profile "Primary values:$vals" \
--natural-language-profile \
--natural-language-profile-detail "no" \
--perspective-amount "extreme" \
--add-high-level-categories \
--verbose  2>&1 | tee -a $SAVE_DIR/log.txt

done
done
done

exit

########################
#### APP: noisy conversation
########################

## System message , 2nd person
#pvq_values_list=(
#  "Hedonism,Stimulation,Self-Direction"
#  "Universalism,Benevolence"
#  "Conformity,Tradition,Security"
#  "Power,Achievement"
#)
#
#message_options=(
#  "System"
##  "User"
#)
#
#person_options=(
#  "2nd"
##  "3rd"
#)
#
##ENGINE="gpt-4-0314"
#ENGINE="gpt-3.5-turbo-0301"
##ENGINE="gpt-3.5-turbo-0613"
##ENGINE="openassistant_rlhf2_llama30b"
##ENGINE="dummy"
#
#PERMUTATIONS=50
#
#for message in "${message_options[@]}"; do
#for person in "${person_options[@]}"; do
#for vals in "${pvq_values_list[@]}"; do
#
#SAVE_DIR="results_test/results_nat_lang_prof_pvq_test_noisy_conv_hello_"$ENGINE"_perm_"$PERMUTATIONS"_"$message"_msg_"$person"_prs"
#mkdir -p $SAVE_DIR
#
#python -u evaluate.py \
#--permutations $PERMUTATIONS \
#--save_dir $SAVE_DIR \
#--engine "$ENGINE" \
#--data_dir data_pvq \
#--experiment_name pvq_test \
#--separator \
#--ntrain 0 \
#$(if [ "$message" == "System" ]; then echo "--system-message"; fi) \
#$(if [ "$person" == "2nd" ]; then echo "--direct-perspective"; fi) \
#--profile "Primary values:$vals" \
#--natural-language-profile \
#--natural-language-profile-detail "no" \
#--perspective-amount "extreme" \
#--add-high-level-categories \
#--add-noisy-conversation \
#--verbose  2>&1 | tee -a $SAVE_DIR/log.txt
#
#done
#done
#done

#######################
### APP: pretend
#######################
#
#pvq_values_list=(
#  "Hedonism,Stimulation,Self-Direction"
#  "Universalism,Benevolence"
#  "Conformity,Tradition,Security"
#  "Power,Achievement"
#)
#
#message_options=(
##  "System"
#  "User"
#)
#
#person_options=(
#  "2nd"
##  "3rd"
#)
#
##ENGINE="gpt-4-0314"
#ENGINE="gpt-3.5-turbo-0301"
##ENGINE="openassistant_rlhf2_llama30b"
##ENGINE="dummy"
#
#PERMUTATIONS=10
#
#for message in "${message_options[@]}"; do
#for person in "${person_options[@]}"; do
#for vals in "${pvq_values_list[@]}"; do
#
#SAVE_DIR="results_neurips/results_nat_lang_prof_pvq_test_pretend_"$ENGINE"_perm_"$PERMUTATIONS"_"$message"_msg_"$person"_prs"
#mkdir -p $SAVE_DIR
#
#python -u evaluate.py \
#--permutations $PERMUTATIONS \
#--save_dir $SAVE_DIR \
#--engine "$ENGINE" \
#--data_dir data_pvq \
#--experiment_name pvq_test \
#--separator \
#--ntrain 0 \
#$(if [ "$message" == "System" ]; then echo "--system-message"; fi) \
#$(if [ "$person" == "2nd" ]; then echo "--direct-perspective"; fi) \
#--profile "Primary values:$vals" \
#--natural-language-profile \
#--natural-language-profile-detail "no" \
#--perspective-amount "extreme" \
#--add-high-level-categories \
#--pretend \
#--verbose  2>&1 | tee -a $SAVE_DIR/log.txt
#
#done
#done
#done


#####################
## 4. Smoothness
#####################


#### PVQ
#############

# Best
# GPT-3.5 (50) S2
# OA (50) U2
# StableVicuna (50) U2
# StableLM (50) U2

#perspective_intensity_list=(
#  "slight"
#  "more"
#  "extreme"
#)
#
## User message , 3nd person (for GPT3.5)
#pvq_values_list=(
#  "Hedonism,Stimulation,Self-Direction"
#  "Universalism,Benevolence"
#  "Conformity,Tradition,Security"
#  "Power,Achievement"
#)
#
#
#
##ENGINE="gpt-4-0314"
##ENGINE="gpt-3.5-turbo-0301"
##ENGINE="openassistant_rlhf2_llama30b"
##ENGINE="stablevicuna"
#ENGINE="stablelm"
##ENGINE="dummy"
#
## Results
## Slight Medium
##GPT35 2.458 & 3.258
##OA 0.804 & 0.867
##StableVicuna 0.194 & 0.328
##StableLM 0.008 & 0.036
#
#PERMUTATIONS=50
#message="User"
#person="2nd"
#
#echo "$ENGINE with $PERMUTATIONS permutations"
#
#for intensity in "${perspective_intensity_list[@]}"; do
#
#echo "Intensity: $intensity"
#
#for vals in "${pvq_values_list[@]}"; do
#
#SAVE_DIR="results_neurips/results_nat_lang_prof_pvq_test_"$ENGINE"_perm_"$PERMUTATIONS"_"$message"_msg_"$person"_prs_intensity_"$intensity""
#mkdir -p $SAVE_DIR
#
#echo "Save dir $SAVE_DIR"
#
#python evaluate.py \
#--permutations $PERMUTATIONS \
#--save_dir $SAVE_DIR \
#--engine "$ENGINE" \
#--data_dir data_pvq \
#--experiment_name pvq_test \
#--separator \
#--ntrain 0 \
#--profile "Primary values:$vals" \
#--natural-language-profile \
#--natural-language-profile-detail "no" \
#--perspective-amount "$intensity" \
#--add-high-level-categories \
#$(if [ "$message" == "System" ]; then echo "--system-message"; fi) \
#$(if [ "$person" == "2nd" ]; then echo "--direct-perspective"; fi) \
#--verbose  2>&1 | tee -a $SAVE_DIR/log.txt
#
#done
#done
