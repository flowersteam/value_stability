#!/bin/bash
#SBATCH -A imi@a100
#SBATCH -C a100
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:2
#SBATCH --array=0-70:5
#SBATCH -o slurm_logs/sb_log_%A_%a.out
#SBATCH -e slurm_logs/sb_log_%A_%a.err

# example: sbatch run_campaign_more_themes.sh Mistral-7B-Instruct-v0.2
##########################################################
# Set the questionnaire and population (using the second command argument)
##########################################################

# Define the configuration based on the experiment_setting
test_tag="pvq"
experiment_name="pvq_test"
data_dir="data_pvq"
population_type="tolkien_characters"
population_config="personas/$population_type/personas.json"

# Print the selected configuration
echo "test_tag=$test_tag"
echo "experiment_name=$experiment_name"
echo "data_dir=$data_dir"
echo "population_type=$population_type"


# Extract parameters: theme and seed
##########################################################
#themes=("grammar" "joke" "poem" "history" "chess" "None")
themes=(
  "grammar"
  "joke"
  "poem"
  "history"
  "chess"

  "bicycle"
  "santa"
  "chord"
  "code"
  "year"

  "cooking"
  "weather"
  "bjj"
  "britney"
  "traveller"

  "None"
)

seed_list=(0 2 4 6 8)

seed_list_len=${#seed_list[@]}


theme_i=$(( SLURM_ARRAY_TASK_ID / $seed_list_len ))
seed_i=$(( SLURM_ARRAY_TASK_ID % $seed_list_len ))

theme="${themes[$theme_i]}"
seed="${seed_list[$seed_i]}"

permute_options_seed="$seed"_"$theme_i"

# Other params
##########################################################
engine="$1"

n_msgs=3

echo "ID:"$SLURM_ARRAY_TASK_ID
echo "Theme:"$theme
echo "Seed:"$seed
echo "Seed str:"$permute_options_seed
echo "Evaluation:$engine:$theme:$permute_options_seed:$n_msgs:$test_tag:$population_type"

# Setup the experiments directories
##########################################################
SUBDIR="stability_default_params_${test_tag}_${population_type}_more_themes/${engine}/seed_${seed}/theme_${theme}"

SAVE_DIR="PLOSONE/results/"$SUBDIR
LOG_DIR="PLOSONE/logs/"$SUBDIR

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