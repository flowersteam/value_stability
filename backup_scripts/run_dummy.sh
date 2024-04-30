##!/bin/bash

##themes=(
##  "grammar"
##  "joke"
##  "poem"
##  "history"
##  "chess"
##)
##
##engines=(
##  "zephyr-7b-beta"
##  "Mistral-7B-v0.1"
##  "Mistral-7B-Instruct-v0.1"
##  "Mistral-7B-Instruct-v0.2"
##  "Mixtral-8x7B-v0.1-4b"
##  "Mixtral-8x7B-Instruct-v0.1-4b"
##  "Mixtral-8x7B-v0.1"
##  "Mixtral-8x7B-Instruct-v0.1"
##)
#
weathers=(
  "rain"
#  "sun"
#  "snow"
#  "thunderstorm"
#  "sandstorm"
#  "blizzard"
)
#
#engines=(
##  "zephyr-7b-beta"
##  "Mistral-7B-v0.1"
##  "Mistral-7B-Instruct-v0.1"
##  "Mistral-7B-Instruct-v0.2"
##  "Mixtral-8x7B-v0.1"
##  "Mixtral-8x7B-Instruct-v0.1"
##  "Mixtral-8x7B-v0.1-4b"
##  "Mixtral-8x7B-Instruct-v0.1-4b"
#)
#
#for engine in "${engines[@]}"; do
#for weather in "${weathers[@]}"; do
#
#python -u evaluate.py \
#--simulated-population-type tolkien_characters \
#--save_dir results_weather_refactor_check/test_weather_"$engine" \
#--engine "$engine" \
#--query-in-reply \
#--data_dir data_pvq \
#--experiment_name pvq_test \
#--ntrain 0 \
#--format chat \
#--no-profile \
#--weather $weather \
#--verbose
#
#done
#done

engines=(
#  "zephyr-7b-beta"
  "llama_2_7b"
)


for engine in "${engines[@]}"; do
for weather in "${weathers[@]}"; do

python -u evaluate.py \
--simulated-population-type tolkien_characters \
--save_dir results_weather_refactor_check/test_weather_"$engine" \
--engine "$engine" \
--query-in-reply \
--system-message \
--data_dir data_pvq \
--experiment_name pvq_test \
--ntrain 0 \
--format chat \
--no-profile \
--weather $weather \
--verbose

done
done
