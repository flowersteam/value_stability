#!/bin/bash

# Total tasks and tasks per run
total_tasks=25
experiment_setting="don"

# Loop over total tasks in steps of tasks_per_run
for ((i=total_tasks-1; i>=0; i-=1)); do
  if [ $((i % 2)) -eq 0 ]; then
    SLURM_ARRAY_TASK_ID=$i bash run_campaign_sim_conv_pvq_seeds.sh 23 $experiment_setting --verbose
  else
    SLURM_ARRAY_TASK_ID=$i bash run_campaign_sim_conv_pvq_seeds.sh 23 $experiment_setting
  fi
done

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

## Loop over total tasks in steps of tasks_per_run
##for ((i=0; i<total_tasks; i+=tasks_per_run)); do
#for ((i=total_tasks-1; i>=0; i-=tasks_per_run)); do
#  # Launch parallel tasks for the current block
#  for ((j=i-tasks_per_run+1; j<=i && j<total_tasks; j++)); do
#    if [ "$j" -eq "$i" ]; then
#      SLURM_ARRAY_TASK_ID=$j bash run_campaign_sim_conv_no_pop.sh 22 --verbose &
#    else
#      SLURM_ARRAY_TASK_ID=$j bash run_campaign_sim_conv_no_pop.sh 22 &
#    fi
#  done
#
#  # Wait for the current block of tasks to finish
#  wait
#  echo "Block $((($tasks_per_run - $i  / $tasks_per_run + 1) - 1)) of $(( (total_tasks + tasks_per_run - 1) / tasks_per_run )) completed."
#done
#
#echo "All tasks have been completed."
#
