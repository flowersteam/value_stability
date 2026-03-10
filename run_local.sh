#!/bin/bash


#SLURM_ARRAY_TASK_ID=1 bash Leaderboard/run_scripts/run_stability_leaderboard_test.sh Mixtral-Large-Instruct-2407

for i in {25..29};
do
#  SLURM_ARRAY_TASK_ID=$i bash run_campaign_more_msgs.sh Mixtral-8x7B-Instruct-v0.1
  SLURM_ARRAY_TASK_ID=$i bash run_campaign_no_pop_more_msgs.sh phi-2
done
exit


#SLURM_ARRAY_TASK_ID=0 bash run_leaderboard_v3.sh gpt-4o-0513
#SLURM_ARRAY_TASK_ID=1 bash run_leaderboard_v3.sh gpt-4o-0513
#SLURM_ARRAY_TASK_ID=2 bash run_leaderboard_v3.sh gpt-4o-0513
#SLURM_ARRAY_TASK_ID=3 bash run_leaderboard_v3.sh gpt-4o-0513
#SLURM_ARRAY_TASK_ID=4 bash run_leaderboard_v3.sh gpt-4o-0513
#SLURM_ARRAY_TASK_ID=5 bash run_leaderboard_v3.sh gpt-4o-0513
#SLURM_ARRAY_TASK_ID=6 bash run_leaderboard_v3.sh gpt-4o-0513
#SLURM_ARRAY_TASK_ID=7 bash run_leaderboard_v3.sh gpt-4o-0513
exit


#for i in {0..6};
#do
#  SLURM_ARRAY_TASK_ID=$i bash run_leaderboard_v3.sh gpt-3.5-turbo-0125
#done
#exit

#SLURM_ARRAY_TASK_ID=0 bash run_leaderboard_v3.sh phi-3-small
#
#for i in {0..5}
#do
#  SLURM_ARRAY_TASK_ID=$i bash run_leaderboard_v3.sh phi-3-
#done
#exit

#for i in $(seq 0 5 70);
#do
##  SLURM_ARRAY_TASK_ID=$i bash run_leaderboard_v2.sh gpt-3.5-turbo-0125
#  SLURM_ARRAY_TASK_ID=$i bash run_campaign_more_themes.sh Mistral-7B-Instruct-v0.2
#done

#exit


#for model in "gpt-3.5-turbo-0125" "gpt-3.5-turbo-1106"
#do
#  for i in {0..24}
#  do
##    SLURM_ARRAY_TASK_ID=$i bash run_campaign_gs.sh $model # GS
#    SLURM_ARRAY_TASK_ID=$i bash run_campaign_seeds.sh $model don # eval
#  done
#done
#
#exit



#
## Total tasks and tasks per run
#total_tasks=25
#experiment_setting="bag"
#
## Loop over total tasks in steps of tasks_per_run
#for ((i=0; i<total_tasks; i+=1)); do
##  if [ $((i % 2)) -eq 0 ]; then
#  if [ $i -eq 0 ]; then
#    SLURM_ARRAY_TASK_ID=$i bash run_campaign_sim_conv_pvq_seeds.sh 23 $experiment_setting --verbose
#  else
#    SLURM_ARRAY_TASK_ID=$i bash run_campaign_sim_conv_pvq_seeds.sh 23 $experiment_setting
#  fi
#done

## Total tasks and tasks per run
#total_tasks=25
#tasks_per_run=1
#
#experiment_setting="don"
#
## Loop over total tasks in steps of tasks_per_run
#for ((i=0; i<total_tasks; i+=tasks_per_run)); do
#  # Launch parallel tasks for the current block
#  for ((j=i; j<i+tasks_per_run && j<total_tasks; j++)); do
#    if [ "$j" -eq "$i" ]; then
#      SLURM_ARRAY_TASK_ID=$j bash run_campaign_sim_conv_pvq_seeds.sh 23 $experiment_setting --verbose &
#    else
#      SLURM_ARRAY_TASK_ID=$j bash run_campaign_sim_conv_pvq_seeds.sh 23 $experiment_setting &
#    fi
#  done
#
#  # Wait for the current block of tasks to finish
#  wait
#  echo "Block $(($i / $tasks_per_run + 1)) of $(( (total_tasks + tasks_per_run - 1) / tasks_per_run )) completed."
#
##  echo "Sleeping 30 secs"
##  sleep 30
#done
#
#echo "All tasks have been completed."

# NO pop - it goes backward  to get shorted msgs results quicker
#############

# Loop over total tasks in steps of tasks_per_run
#for ((i=0; i<total_tasks; i+=tasks_per_run)); do

# backwards
#for ((i=total_tasks-1; i>=0; i-=tasks_per_run)); do
# 15 - 20 : just msgs =3

#for ((i=3; i<25; i+=5)); do # Launch parallel tasks for the current block
#  SLURM_ARRAY_TASK_ID=$i bash run_campaign_sim_conv_no_pop.sh 23
#
#done
#
#echo "All tasks have been completed."
#
