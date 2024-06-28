
RESULTS_DIR="./Leaderboard/results/stability_leaderboard"
ANALYSIS_RESULTS_DIR="./Leaderboard/data_analysis/analysis_results"

mkdir -p $ANALYSIS_RESULTS_DIR

models=(
 "phi-3-mini-128k-instruct"
 "phi-3-medium-128k-instruct"
 "Mistral-7B-Instruct-v0.1"
 "Mistral-7B-Instruct-v0.2"
 "Mistral-7B-Instruct-v0.3"
 "Mixtral-8x7B-Instruct-v0.1"
 "Mixtral-8x22B-Instruct-v0.1"
 "command_r_plus"
 "llama_3_8b_instruct"
 "llama_3_70b_instruct"
 "Qwen2-7B-Instruct"
 "Qwen2-72B-Instruct"
 "gpt-3.5-turbo-0125"
 "gpt-4o-0513"
)

for model in "${models[@]}"
do
  ANALYSIS_RESULTS_JSON_PATH=$ANALYSIS_RESULTS_DIR/$model.json
  python ./visualization_scripts/data_analysis.py  $RESULTS_DIR/$model/* --structure --cronbach-alpha --cfa --ips --result-json-savepath $ANALYSIS_RESULTS_JSON_PATH  &> /dev/null

  # display the Rank-Order stability
  echo -n "$model : "
  python -c "import json; print(json.load(open('"$ANALYSIS_RESULTS_JSON_PATH"'))['Rank-Order'])"
done