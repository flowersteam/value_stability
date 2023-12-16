##!/bin/bash
#
#PERMUTATIONS=1
##ENGINE="gpt-3.5-turbo-0301"
##ENGINE="gpt-3.5-turbo-0613"
#
## College
#SAVE_DIR="results_iclr/results_mmlu_college_test_format_"$ENGINE"_perm_"$PERMUTATIONS"_format"
#mkdir -p $SAVE_DIR
#
#format_options=(
#  "chat"
#  "code_py"
#  "code_cpp"
#  "conf_toml"
#  "latex"
#)
#
#for format in "${format_options[@]}"; do
#
#python -u evaluate.py \
#--permutations $PERMUTATIONS \
#--save_dir $SAVE_DIR \
#--engine $ENGINE \
#--data_dir data_mmlu \
#--experiment_name mmlu_college \
#--ntrain 0 \
#--estimate-gpt-tokens \
#--format $format \
#--no-profile \
#--verbose
#
#done


# High school

PERMUTATIONS=1
ENGINE="gpt-3.5-turbo-0301"
#ENGINE="gpt-3.5-turbo-0613"

SAVE_DIR="results_iclr/results_mmlu_high_school_test_format_"$ENGINE"_perm_"$PERMUTATIONS"_format"
mkdir -p $SAVE_DIR

format_options=(
  "chat"
  "code_py"
  "code_cpp"
  "conf_toml"
  "latex"
)

for format in "${format_options[@]}"; do

python -u evaluate.py \
--permutations $PERMUTATIONS \
--save_dir $SAVE_DIR \
--engine $ENGINE \
--data_dir data_mmlu \
--experiment_name mmlu_high_school \
--ntrain 0 \
--estimate-gpt-tokens \
--format $format \
--no-profile \
--verbose

done
