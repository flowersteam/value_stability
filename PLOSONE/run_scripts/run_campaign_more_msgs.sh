#!/bin/bash
#SBATCH -A imi@a100
#SBATCH -C a100
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:2
#SBATCH --array=0-34
#SBATCH -o slurm_logs/sb_log_%A_%a.out
#SBATCH -e slurm_logs/sb_log_%A_%a.err

# example: sbatch run_campaign_more_msgs.sh
##########################################################
# Set the questionnaire and population (using the second command argument)
##########################################################

# Define the configuration based on the experiment_setting
test_tag="pvq"
experiment_name="pvq_test"
data_dir="data_pvq"
population_type="tolkien_characters"
population_config="personas/$population_type/personas.json"

#####################################################

# extract theme n_msgs and n_msgs

themes=("grammar" "joke" "poem" "history" "chess")
themes_len=${#themes[@]}

n_msgs_list=(3 7 11 19 27 35 43)
n_msgs_len=${#n_msgs_list[@]}


echo "ID:"$SLURM_ARRAY_TASK_ID

msgs_i=$(( SLURM_ARRAY_TASK_ID / $themes_len ))
theme_i=$(( SLURM_ARRAY_TASK_ID % $themes_len ))

theme="${themes[$theme_i]}"
n_msgs="${n_msgs_list[$msgs_i]}"

seed="1"

echo "Msg_i:"$n_msgs
echo "Theme_i:"$theme
echo "Seed_i:"$seed

permute_options_seed="$seed"_"$theme_i"

# Other params
##########################################################
engine="Mixtral-8x7B-Instruct-v0.1"

echo "ID:"$SLURM_ARRAY_TASK_ID
echo "Theme:"$theme
echo "Seed:"$seed
echo "Seed str:"$permute_options_seed
echo "Evaluation:$engine:$theme:$permute_options_seed:$n_msgs:$test_tag:$population_type"

SUBDIR="stability_default_params_"$test_tag"_"$population_type"_more_msgs/"$engine"/"$n_msgs"_msgs/"$seed"_seed/results_sim_conv_"$population_type"_"$theme

SAVE_DIR="PLOSONE/results/"$SUBDIR
LOG_DIR="PLOSONE/logs/"$SUBDIR

echo "SAVEDIR:"$SAVE_DIR

# Start the experiment
##########################################################
mkdir -p $LOG_DIR

echo "SLURM_JOB_ID: "$SLURM_JOB_ID"_"$SLURM_ARRAY_TASK_ID | tee -a $LOG_DIR/log_$permute_options_seed.txt

source $HOME/.bashrc

module load python/3.10.4
conda activate llm_stability_441

python -u evaluate.py \
  --simulated-population-config $population_config \
  --simulated-conversation-theme $theme \
  --interlocutor-knows-persona \
  --simulated-conversation-n-messages $n_msgs \
  --permute-options-seed "$permute_options_seed" \
  --save_dir $SAVE_DIR \
  --engine "$engine" \
  --data_dir data/$data_dir \
  --experiment_name $experiment_name \
  --pvq-version "pvq_auto" \
  --assert-params \
  --verbose  2>&1 | tee -a $LOG_DIR/log_$permute_options_seed.txt