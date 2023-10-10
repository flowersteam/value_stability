##!/bin/bash

#"chat"
#"code_py"
#"code_cpp"
#"conf_toml"
#"latex"


#python -u evaluate.py \
#--permutations 50 \
#--save_dir results_test/test \
#--engine dummy \
#--data_dir data_big5_50 \
#--experiment_name big5_test \
#--ntrain 0 \
#--estimate-gpt-tokens \
#--format latex \
#--no-profile \
#--verbose
#exit
#
#
#SAVE_DIR="results_iclr/results_pvq_test_sim_conv_"$ENGINE"_perm_"$PERMUTATIONS"_theme"
#mkdir -p $SAVE_DIR
## conversation
python -u evaluate.py \
--permutations 50 \
--save_dir results_test/test \
--engine gpt-3.5-turbo-0613 \
--data_dir data_pvq \
--experiment_name test_pvq \
--ntrain 0 \
--format chat \
--no-profile \
--simulate-conversation-theme "chess" \
--system-message \
--verbose
#
#done

########## Themes - with set values
## conversation
#python -u evaluate.py \
#--permutations 50 \
#--save_dir results_test/test \
#--engine gpt-3.5-turbo-0613 \
#--data_dir data_pvq \
#--experiment_name test_pvq \
#--separator \
#--ntrain 0 \
#--format chat \
#--simulate-conversation-theme $theme \
#--system-message \
#--direct-perspective \
#--verbose


#themes=(
##  "poem"
##  "joke"
##  "history"
##  "grammar"
#  "chess"
#)

#vals="Universalism,Benevolence"
#
#for theme in "${themes[@]}"; do
#
## conversation
#python -u evaluate.py \
#--permutations 50 \
#--save_dir results_test/test \
#--engine gpt-3.5-turbo-0613 \
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
#--verbose
#
#done

#python evaluate.py \
#--permutations 1 \
#--save_dir results/test \
#--engine gpt-3.5-turbo-0613 \
#--data_dir data_pvq \
#--experiment_name test_pvq \
#--ntrain 0 \
#--verbose \
#--estimate-gpt-tokens \
#--music-expert-genre  "jazz" \
#--system-message \
#--add-noisy-conversation \

#--engine openassistant_rlhf2_llama30b \

#python evaluate.py \
#--permutations 10 \
#--ntrain 0 \
#--data_dir data_pvq \
#--save_dir results/test \
#--engine dummy \
#--experiment_name test_pvq \
#--perspective-amount "more" \
#--verbose \
#--separator \
#--profile "$profile" \
#--natural-language-profile \
#--natural-language-profile-detail "no" \
#--add-high-level-categories \
#--direct-perspective \
#--system-message

#--add-noisy-conversation

#--music-expert-genre  "hip-hop" \
#--wiki-context --separator
#--mcq-context


#--system-message \
#--estimate-gpt-tokens \
#--add-high-level-categories \

#done



## LLaMa
#torchrun --nproc_per_node 1 evaluate.py \
#--permutations 1 \
#--ntrain 0 \
#--data_dir data_pvq \
#--engine dummy \
#--save_dir results_test/test_dummy \
#--experiment_name pvq_test \
#--perspective-amount "extreme" \
#--profile "$profile" \
#--separator \
#--natural-language-profile \
#--natural-language-profile-detail "no" \
#--add-high-level-categories \
#--verbose
#
##--direct-perspective \
#
#done
#
#
