#!/bin/bash
#SBATCH -A imi@a100
#SBATCH -C a100
#SBATCH --time=06:30:00
#SBATCH --gres=gpu:2
#SBATCH --array=0-24 # themes x n_msg -> 5x5 (no default profile, only contexts)
#SBATCH -o slurm_logs/sb_log_%A_%a.out
#SBATCH -e slurm_logs/sb_log_%A_%a.err
##SBATCH --qos=qos_gpu-dev

##########################################################
# Set the questionnaire and population (using the second command argument)
##########################################################

test_tag="pvq"
experiment_name="pvq_test"
data_dir="data_pvq"
population_type="permutations"

# Print the selected configuration
echo "test_tag=$test_tag"
echo "experiment_name=$experiment_name"
echo "data_dir=$data_dir"
echo "population_type=$population_type"


# Extract parameters: theme and seed
##########################################################
themes=("grammar" "joke" "poem" "history" "chess" "None")

n_msgs_list=(9 7 5 3 1) # 5
n_msgs_len=${#n_msgs_list[@]}

theme_i=$(( SLURM_ARRAY_TASK_ID / $n_msgs_len ))
msgs_i=$(( SLURM_ARRAY_TASK_ID % $n_msgs_len ))

theme="${themes[$theme_i]}"
n_msgs="${n_msgs_list[$msgs_i]}"

permute_options_seed=$theme_i

# Other params
##########################################################
engine="$1"

echo "ID:"$SLURM_ARRAY_TASK_ID
echo "Theme:"$theme
echo "Seed:"$seed
echo "Seed str:"$permute_options_seed
echo "Evaluation:$engine:$theme:$permute_options_seed:$n_msgs:$test_tag:$population_type"

# Setup the experiments directories
##########################################################
SUBDIR="stability_default_params_${test_tag}_${population_type}_msgs/${engine}/${n_msgs}_msgs/${seed}_seed/theme_${theme}"

SAVE_DIR="results/"$SUBDIR
LOG_DIR="logs/"$SUBDIR

# Start the experiment
##########################################################
mkdir -p $LOG_DIR

source $HOME/.bashrc

## define the conda env to use
case "$engine" in
    phi-1|phi-2|Qwen1.5*|llama_3*|command_r_plus*|Mixtral-8x22B*)
        conda activate llm_stability_phi
        ;;
    *)
        conda activate llm_stability
        ;;
esac

echo "SLURM_JOB_ID: "$SLURM_JOB_ID"_"$SLURM_ARRAY_TASK_ID | tee -a $LOG_DIR/log_$permute_options_seed.txt

python -u evaluate.py \
  --simulated-population-type $population_type \
  --simulated-conversation-theme $theme \
  --simulated-conversation-n-messages $n_msgs \
  --permute-options \
  --permute-options-seed "$permute_options_seed" \
  --save_dir $SAVE_DIR \
  --engine "$engine" \
  --data_dir data/$data_dir \
  --experiment_name $experiment_name \
  --pvq-version "pvq_auto" \
  --azure-openai \
  --assert-params \
  --verbose  2>&1 | tee -a $LOG_DIR/log_$permute_options_seed.txt