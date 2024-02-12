##!/bin/bash

engines=(
  "dummy"
#  "zephyr-7b-beta"
#  "Mistral-7B-v0.1"
#  "Mistral-7B-Instruct-v0.1"
#  "Mistral-7B-Instruct-v0.2"
#  "Mixtral-8x7B-Instruct-v0.1"
#  "Mixtral-8x7B-Instruct-v0.1-4b"
#  "Mixtral-8x7B-v0.1-4b"
#  "Mixtral-8x7B-v0.1"
#  "Mixtral-8x7B-Instruct-v0.1"
#  "zephyr-7b-beta"
#  "llama_2_7b_chat" # 2 gpu
#  "falcon-7b"
#  "phi-2"
#  "phi-1.5"
#  "phi-1"
#  "Qwen-72B"
#  "Qwen-14B"
#  "Qwen-7B"
#  "Qwen-72B-Chat"
)

for engine in "${engines[@]}"; do

#--simulated-population-type permutations \
#--permutations 50 \


#--simulated-population-type famous_people \

#--simulated-population-type permutations \
#--permutations 50 \

python -u evaluate.py \
--simulated-population-type tolkien_characters \
--simulate-conversation-theme "chess" \
--simulated-human-knows-persona \
--simulated-conversation-n-messages 3 \
--permute-options \
--permute-options-seed "5" \
--format chat \
--save_dir results/test/test \
--engine $engine \
--query-in-reply \
--data_dir data/data_wvs_svas \
--experiment_name wvs_svas_test \
--pvq-version "pvq_auto" \
--no-profile \
--direct-perspective \
--base-model-template \
--system-message \
--verbose


done

#--data_dir data/data_wvs_svas \
#--experiment_name wvs_svas_test \

#--data_dir data/data_pvq \
#--experiment_name pvq_test \

#--data_dir data/data_tolkien_donation \
#--experiment_name tolkien_donation_test \

#--data_dir data/data_tolkien_ultimatum \
#--experiment_name tolkien_ultimatum_test \

#--data_dir data/data_regular_ultimatum \
#--experiment_name regular_ultimatum_test \

#--data_dir data/data_tolkien_good_bad \
#--experiment_name tolkien_good_bad_test \
