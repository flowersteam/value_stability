

RESULTS_DIR="./Leaderboard/results/stability_leaderboard"
ANALYSIS_RESULTS_DIR="./Leaderboard/data_analysis/analysis_results"

mkdir -p $ANALYSIS_RESULTS_DIR

models=(
#  "GLM-4-32B-0414"

#  "Qwen3-235B-A22B-FP8"
#  "Qwen3-32B-A3B"
#  "Qwen3-32B"
#  "Qwen3-8B"
#  "Qwen3-4B"

#    "reka-flash-3"
#
#    "DeepSeek-V3-0324"
#    "DeepSeek-V3-0324_user"
#    "DeepSeek-R1"
#
#    "gemma-3-27b-it"

#    "Llama-4-Scout-17B-16E-Instruct"
#    "Llama-3.3-70B-Instruct"
#    "Llama-3.1-70B-Instruct"
#    "Llama-3.1-Nemotron-70B-Instruct"
#    "Llama-3.1-8B-Instruct"
#
#    "Llama-3.2-3B-Instruct"
#    "Llama-3.2-1B-Instruct"
#
#    "Mistral-Large-Instruct-2411"
#    "Mistral-Large-Instruct-2407"
#    "Mistral-Nemo-Instruct-2407"
#    "Mistral-Small-3.1-24B-Instruct-2503"
#    "Mistral-7B-Instruct-v0.2"
#    "Mixtral-8x7B-Instruct-v0.1"
#
#    "QwQ-32B"
#
#    "Qwen2.5-VL-72B-Instruct"
    "Qwen2.5-VL-32B-Instruct"
#    "Qwen2.5-VL-7B-Instruct"
#    "Qwen2.5-VL-3B-Instruct"
#    "Qwen2.5-72B-Instruct"
#    "Qwen2.5-32B-Instruct"
#    "Qwen2.5-14B-Instruct-1M"
#
#    "phi-4"
#    "phi-3-medium-128k-instruct"
#
#    "Dracarys2-72B-Instruct"
#
#    "Nautilus-70B-v0.1"
#    "Cydonia-22B-v1.2"
#    "Ministrations-8B-v1"
#
#    "dummy"
)


for model in "${models[@]}"
do
  echo -n "$model : "

#  metrics for the leaderboard
  ANALYSIS_RESULTS_JSON_PATH=$ANALYSIS_RESULTS_DIR/$model.json
  python ./visualization_scripts/data_analysis.py  $RESULTS_DIR/$model/* --plot-save --plot-structure --plot-ranks --plot-matrix --structure --cronbach-alpha --cfa --result-json-savepath $ANALYSIS_RESULTS_JSON_PATH

  # display the Rank-Order stability
  echo ""
  python -c "import json; print('\tRO:'+str(json.load(open('"$ANALYSIS_RESULTS_JSON_PATH"'))['Rank-Order']))"
#  python -c "import json; print('\tCFI:'+str(json.load(open('"$ANALYSIS_RESULTS_JSON_PATH"'))['CFI']))"

done