##!/bin/bash


###########################################
# 1. Implied values (directly / indirectly)
###########################################

# LoTR characters
##################

# 2ps (direct pers) ; System message

#ENGINE="gpt-4-0314"
##ENGINE="openassistant_rlhf2_llama30b"
##ENGINE="stablevicuna"
##ENGINE="stablelm"
##ENGINE="dummy"
##ENGINE="gpt-3.5-turbo-0301"
#
#lotr_characters=(
#  "Gandalf"
#  "Frodo"
#  "Sauron"
#  "Aragorn"
#  "Pippin"
#)
#
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

### Music AI Experts
####################
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
#PERMUTATIONS=1
#ENGINE="gpt-4-0314"
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

#### Sys: 2nd 3rd | Usr: 2nd 3rd
# GPT-4 (5)  & 2.309 & 2.439 & 2.095 & 2.226 -> not fair
# GPT-3.5 (10) & 3.162 & 2.715 & 3.122 & 2.729 -> fair
# GPT-3.5 (50) $ 3.403 $ 2.803 & 3.202 & 2.82
# OA (50) & 0.619 & 0.698 & 0.979 & 0.647
# StableVicuna (50)  & n/a & n/a & 0.328 & 0.168
# StableLM (50) & -0.029 & -0.009 & 0.029 & -0.001

# System message , 2nd person
pvq_values_list=(
  "Hedonism,Stimulation,Self-Direction"
  "Universalism,Benevolence"
  "Conformity,Tradition,Security"
  "Power,Achievement"
)

message_options=(
  "System"
#  "User"
)

person_options=(
  "2nd"
#  "3rd"
)

ENGINE="gpt-4-0314"
#ENGINE="gpt-3.5-turbo-0301"
#ENGINE="openassistant_rlhf2_llama30b"
#ENGINE="dummy"

PERMUTATIONS=1

for message in "${message_options[@]}"; do
for person in "${person_options[@]}"; do
for vals in "${pvq_values_list[@]}"; do

SAVE_DIR="results_neurips/results_nat_lang_prof_pvq_test_"$ENGINE"_perm_"$PERMUTATIONS"_"$message"_msg_"$person"_prs"
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
#  "medium"
##  "extreme"
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
##ENGINE="stablelm"
#ENGINE="dummy"
#
## Results
## Slight Medium
##GPT35 2.458 & 3.258
##OA 0.804 & 0.867
##StableVicuna 0.194 & 0.328
##StableLM 0.008 & 0.036
#
#PERMUTATIONS=50
#message="System"
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
