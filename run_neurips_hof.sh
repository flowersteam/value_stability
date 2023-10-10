##!/bin/bash

####################################
### 3. System/User 2ne/3rd person
####################################

#### HOFSTEDE
########################

##### Sys: 2nd 3rd | Usr: 2nd 3rd
# GPT-4 (5) 172.333 & 177.033 & 149.5 & 188.4
# GPT-3.5 (5) 69.033 & 102.533 & 128.5 & 122.733
# GPT-3.5 (50) 80.59 & 99.623 & 124.99 & 110.713
# OA (50)  4.63 & 13.79 & 20.57 & 24.083
# StableVicuna (50)  & n/a & n/a & -1.76 & 3.01
# StableLM (50) & -2.367 & 2.743 & 1.943 & 2.287

## System message , 2nd person
hof_values_list=(
  "Power Distance"
  "Masculinity"
  "Uncertainty Avoidance"
  "Long-Term Orientation"
  "Indulgence"
  "Individualism"
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
ENGINE="dummy"

ENGINE="gpt-3.5-turbo-instruct-0914"

PERMUTATIONS=50

for message in "${message_options[@]}"; do
for person in "${person_options[@]}"; do
for vals in "${hof_values_list[@]}"; do

SAVE_DIR="results_neurips/results_nat_lang_prof_hofstede_test_"$ENGINE"_perm_"$PERMUTATIONS"_"$message"_msg_"$person"_prs"
mkdir -p $SAVE_DIR

python -u evaluate.py \
--permutations $PERMUTATIONS \
--save_dir $SAVE_DIR \
--engine "$ENGINE" \
--data_dir data_hofstede \
--experiment_name hofstede_test \
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



#### HOFSTEDE
#############

# Best
# GPT-4 (5)
# GPT-3.5 (10) S2
# GPT-3.5 (50)
# OA (50) U2
# StableVicuna (50) U2
# StableLM (50) U2

#perspective_intensity_list=(
#  "slight"
#  "medium"
#  "more"
#  "extreme"
#)
#
## User message , 3nd person (for GPT3.5)
#hof_values_list=(
#  "Power Distance"
#  "Masculinity"
#  "Uncertainty Avoidance"
#  "Long-Term Orientation"
#  "Indulgence"
#  "Individualism"
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
##OA
##StableVicuna & 8.08 & 4.94
##StableLM & 0.723 & -0.933
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
#for vals in "${hof_values_list[@]}"; do
#
#SAVE_DIR="results_neurips/results_nat_lang_prof_hofstede_test_"$ENGINE"_perm_"$PERMUTATIONS"_"$message"_msg_"$person"_prs_intensity_"$intensity""
#mkdir -p $SAVE_DIR
#
#echo "Save dir $SAVE_DIR"
#
#python evaluate.py \
#--permutations $PERMUTATIONS \
#--save_dir $SAVE_DIR \
#--engine "$ENGINE" \
#--data_dir data_hofstede \
#--experiment_name hofstede_test \
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
