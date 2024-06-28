#!/bin/bash
## V100 -jz
##SBATCH -A imi@v100
##SBATCH -C v100-32g
##SBATCH --time=09:59:00
##SBATCH --gres=gpu:3
### A100 -jz
#SBATCH -A imi@a100
#SBATCH -C a100
#SBATCH --time=7:00:00
#SBATCH --gres=gpu:3
### Adastra
##SBATCH --time=6:00:00 # maximum execution time (HH:MM:SS)
##SBATCH --account=iso1996
##SBATCH -C MI250
###SBATCH --exclusive
##SBATCH --gres=gpu:1
##SBATCH --cpus-per-task=8
##SBATCH --hint=nomultithread
##SBATCH --ntasks-per-node=1
##SBATCH --nodes=1
## other params
#SBATCH --array=0-8 # all
#SBATCH -o slurm_logs/log_%A_%a.out
#SBATCH -e slurm_logs/log_%A_%a.err

##########################################################
# Set the questionnaire and population (using the second command argument)
##########################################################

# Define the configuration based on the experiment_setting
test_tag="pvq"
experiment_name="pvq_test"
data_dir="data_pvq"
population_config="personas/real_world_people/personas.json"

eval_i=$SLURM_ARRAY_TASK_ID


# Msgs
####################
n_msgs=3

# Conversation topics chunk and interlocutor personas
##########################################################
# 0-> no conversation (human inter.)
# 1-5 -> reddit chunks (human inter.)
# 6 -> chess (personas inter chunk 0)
# 7 -> grammar (personas inter chunk 1)

if [ $eval_i -eq 0 ]; then
  # no conversation
  chunk_i="no_conv"
  theme="None"
  interlocutors="human"

elif [ $eval_i -eq 6 ]; then
  # classic themes (chess), interlocutors chunk 0
  chunk_i="chess_"$[eval_i - 6]
  theme="chess"

  interlocutors_chunk=0
  interlocutors="./interlocutors/real_world_people/chunk_"$interlocutors_chunk"/interlocutors.json"

elif [ $eval_i -eq 7 ]; then
  # classic themes (grammar), interlocutors chunk 1
  chunk_i="grammar_"$[eval_i - 6]
  theme="grammar"

  interlocutors_chunk=1
  interlocutors="./interlocutors/real_world_people/chunk_"$interlocutors_chunk"/interlocutors.json"

elif [ $eval_i -eq 8 ]; then

  test_tag="svs"
  experiment_name="svs_test"
  data_dir="data_svs"

  # no conversation
  chunk_i="svs_no_conv"
  theme="None"
  interlocutors="human"

elif [ $eval_i -ge 9 ]; then
  # undefined
  echo "Undefined eval_i:$eval_i"
  exit

else
  # 1 - 5 -> chunks: 0 - 4
  # reddit chunks
  chunk_i=$[eval_i - 1]
  theme="contexts/leaderboard_reddit_chunks/chunk_"$chunk_i".jsonl"

  interlocutors="human"
fi


# Permutations seed
##########################################################
permute_options_seed="leaderboard_"$chunk_i

# Model
####################
engine="$1"
model_config="./models/leaderboard_configs/$engine.json"

if [ ! -f $model_config ]; then
  echo "No model config found at $model_config"
  exit
fi


# Setup the experiment directories
####################################t######################

SUBDIR="stability_leaderboard/${engine}/chunk_${chunk_i}"
SAVE_DIR="Leaderboard/results/"$SUBDIR
LOG_DIR="Leaderboard/logs/"$SUBDIR

echo "SAVE DIR: "$SAVE_DIR

echo "Evaluation"
echo "ID:"$SLURM_ARRAY_TASK_ID
echo "engine:"$engine
echo "theme:"$theme
echo "permute_options_seed:"$permute_options_seed
echo "n_msgs:"$n_msgs
echo "population_config:"$population_config
echo "interlocutors:"$interlocutors
echo "test_tag:"$test_tag
echo "experiment_name:$experiment_name"
echo "data_dir:$data_dir"
echo "eval_i:$eval_i"
echo "chunk_i:$chunk_i"
echo "savedir:$SAVE_DIR"
echo "logdir:$LOG_DIR"


# Start the experiment
##########################################################
mkdir -p $LOG_DIR

echo "SLURM_JOB_ID: "$SLURM_JOB_ID"_"$SLURM_ARRAY_TASK_ID | tee -a $LOG_DIR/log_$permute_options_seed.txt


source $HOME/.bashrc
## define the conda env to use

# jz
module load python/3.10.4
conda activate llm_stability_441

## adastra
#module load conda
#conda activate llm_stability_441

python -u evaluate.py \
  --simulated-population-config $population_config \
  --simulated-conversation-theme $theme \
  --simulated-conversation-n-messages $n_msgs \
  --permute-options-seed "$permute_options_seed" \
  --save_dir $SAVE_DIR \
  --interlocutors $interlocutors \
  --model-config-path $model_config \
  --data_dir data/$data_dir \
  --experiment_name $experiment_name \
  --pvq-version "pvq_auto" \
  --assert-params \
  --verbose  2>&1 | tee -a $LOG_DIR/log_$permute_options_seed.txt