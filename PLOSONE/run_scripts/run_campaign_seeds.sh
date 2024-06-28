#!/bin/bash
#SBATCH -A imi@a100
#SBATCH -C a100
#SBATCH --time=06:30:00
#SBATCH --gres=gpu:2
#SBATCH --array=0-24 # themes x n_seeds -> 6x5 (0-24 wo None, 0-29 for all)
#SBATCH -o slurm_logs/sb_log_%A_%a.out
#SBATCH -e slurm_logs/sb_log_%A_%a.err
##SBATCH --qos=qos_gpu-dev

##########################################################
# Set the questionnaire and population (using the second command argument)
##########################################################

experiment_setting=$2

# Define the configuration based on the experiment_setting
case "$experiment_setting" in
  pvq_tolk)
    test_tag="pvq"
    experiment_name="pvq_test"
    data_dir="data_pvq"
    population_type="tolkien_characters"
    population_config="personas/tolkien_characters/personas.json"
    ;;
  pvq_fam)
    test_tag="pvq"
    experiment_name="pvq_test"
    data_dir="data_pvq"
    population_type="famous_people"
    population_config="personas/famous_people/personas.json"
    ;;
  don)
    test_tag="tolkien_donation"
    experiment_name="tolkien_donation_test"
    data_dir="data_tolkien_donation"
    population_type="tolkien_characters"
    population_config="personas/tolkien_characters/personas.json"
    ;;
  bag)
    test_tag="tolkien_bag"
    experiment_name="tolkien_bag_test"
    data_dir="data_tolkien_bag"
    population_type="tolkien_characters"
    population_config="personas/tolkien_characters/personas.json"
    ;;
  religion)
    test_tag="religion"
    experiment_name="religion_test"
    data_dir="data_religion"
    population_type="famous_people"
    population_config="personas/famous_people/personas.json"
    ;;
  no_pop)
    test_tag="pvq"
    experiment_name="pvq_test"
    data_dir="data_pvq"
    population_type="permutations"
    population_config="permutations"
    ;;
  *)
    echo "Invalid experiment_setting. Please use one of the following: pvq_tolk, pvq_fam, don, bag, religion."
    exit 1
    ;;
esac

# Print the selected configuration
echo "test_tag=$test_tag"
echo "experiment_name=$experiment_name"
echo "data_dir=$data_dir"
echo "population_config=$population_config"


# Extract parameters: theme and seed
##########################################################
themes=("grammar" "joke" "poem" "history" "chess" "None")
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
echo "Evaluation:$engine:$theme:$permute_options_seed:$n_msgs:$test_tag:$population_config"

# Setup the experiments directories
##########################################################
SUBDIR="stability_default_params_${test_tag}_${population_type}/${engine}/seed_${seed}/theme_${theme}"

SAVE_DIR="PLOSONE/test_results/"$SUBDIR
LOG_DIR="PLOSONE/test_logs/"$SUBDIR

# Start the experiment
##########################################################
mkdir -p $LOG_DIR

echo "SLURM_JOB_ID: "$SLURM_JOB_ID"_"$SLURM_ARRAY_TASK_ID | tee -a $LOG_DIR/log_$permute_options_seed.txt

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
#conda activate my_env


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