##!/bin/bash

####################################
### 3. System/User 2ne/3rd person
####################################

#### Big 5
########################

##### Sys: 2nd 3rd | Usr: 2nd 3rd
# GPT-4 (5) & 17.49 & 17.89 & 17.73 & 18.39
# GPT-3.5 (5) & 17.8 & 18.98 & 19.13 & 19.41
# GPT-3.5 (50)
# OA (50) & 3.098 & 2.429 & 2.859 & 4.927
# StableVicuna (50) & n/a & n/a & 2.15 & 3.35
# StableLM (50) & 0.0 & 0.003 & 0.205 & -0.041

## System message , 2nd person
big5_values_list=(
  "Neuroticism"
  "Extraversion"
  "Openness to Experience"
  "Agreeableness"
  "Conscientiousness"
)

message_options=(
#  "System"
  "User"
)

person_options=(
  "2nd"
  "3rd"
)

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

#ENGINE="ada"
#ENGINE="babbage"
#ENGINE="curie"
#ENGINE="text-davinci-003"
#ENGINE="dummy"


PERMUTATIONS=50

for message in "${message_options[@]}"; do
for person in "${person_options[@]}"; do
for vals in "${big5_values_list[@]}"; do

SAVE_DIR="results_neurips/results_nat_lang_prof_big5_test_"$ENGINE"_perm_"$PERMUTATIONS"_"$message"_msg_"$person"_prs"
mkdir -p $SAVE_DIR

python -u evaluate.py \
--permutations $PERMUTATIONS \
--save_dir $SAVE_DIR \
--engine "$ENGINE" \
--data_dir data_big5_50 \
--experiment_name big5_test \
--separator \
--ntrain 0 \
$(if [ "$message" == "System" ]; then echo "--system-message"; fi) \
$(if [ "$person" == "2nd" ]; then echo "--direct-perspective"; fi) \
--profile "Primary values:$vals" \
--natural-language-profile \
--natural-language-profile-detail "no" \
--perspective-amount "extreme" \
--verbose  2>&1 | tee -a $SAVE_DIR/log.txt

done
done
done


#####################
## 4. Smoothness
#####################

### Big5
############

# Best
# GPT-4 (5)
# GPT-3.5 (10) S2
# GPT-3.5 (50)
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
#big5_values_list=(
#  "Neuroticism"
#  "Extraversion"
#  "Openness to Experience"
#  "Agreeableness"
#  "Conscientiousness"
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
#message="User"
#person="3rd"
#
## Results
## Slight Medium
##GPT4
##GPT35
##OA & 3.223
##StableVicuna & 1.695 & 3.179 &
##StableLM & -0.06 & 0.003 &
##
#
#PERMUTATIONS=50
#
#echo "$ENGINE with $PERMUTATIONS permutations"
#
#for intensity in "${perspective_intensity_list[@]}"; do
#
#echo "Intensity: $intensity"
#
#for vals in "${big5_values_list[@]}"; do
#
#SAVE_DIR="results_neurips/results_nat_lang_prof_big5_test_"$ENGINE"_perm_"$PERMUTATIONS"_"$message"_msg_"$person"_prs_intensity_"$intensity""
#mkdir -p $SAVE_DIR
#
#echo "Save dir $SAVE_DIR"
#
#python evaluate.py \
#--permutations $PERMUTATIONS \
#--save_dir $SAVE_DIR \
#--engine "$ENGINE" \
#--data_dir data_big5_50 \
#--experiment_name big5_test \
#--separator \
#--ntrain 0 \
#--profile "Primary values:$vals" \
#--natural-language-profile \
#--natural-language-profile-detail "no" \
#$(if [ "$message" == "System" ]; then echo "--system-message"; fi) \
#$(if [ "$person" == "2nd" ]; then echo "--direct-perspective"; fi) \
#--perspective-amount "$intensity" \
#--verbose  2>&1 | tee -a $SAVE_DIR/log.txt
#
#done
#done
