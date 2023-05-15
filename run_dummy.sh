##!/bin/bash

#values_list=(
#  "Power Distance"
#  "Masculinity"
#  "Uncertainty Avoidance"
#  "Long-Term Orientation"
#  "Indulgence"
#  "Individualism"
#)

values_list=(
  "Neuroticism"
  "Extraversion"
  "Openness to Experience"
  "Agreeableness"
  "Conscientiousness"
)

for val in "${values_list[@]}"; do

profile="Primary values:$val"

#openassistant_rlhf2_llama30b

#torchrun --nproc_per_node 4 \
#--engine llama_30B \

python evaluate.py \
--permutations 1 \
--ntrain 0 \
--data_dir data_big5_100 \
--save_dir results/results_big5_100_test_gpt4 \
--engine gpt-4-0314 \
--experiment_name big5_test \
--perspective-amount "extreme" \
--profile "$profile" \
--separator \
--natural-language-profile \
--natural-language-profile-detail "no" \
--estimate-gpt-tokens \
--direct-perspective \
--system-message \
--verbose

#--add-high-level-categories \

done
