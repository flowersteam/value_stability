import json;
import matplotlib.pyplot as plt
from plot_utils import *

# metric = 'Rank-Order'
# metric = 'Structure_correlation'
# metric = 'Cronbach_alpha'

# metric = 'CFI'
# metric = 'TLI'
# metric = 'SRMR'
metric = 'RMSEA'

print(f"Ranking models cardinally based on {metric}.")

models = [
    "phi-3-mini-128k-instruct",
    "phi-3-medium-128k-instruct",
    "Mistral-7B-Instruct-v0.1",
    "Mistral-7B-Instruct-v0.2",
    "Mistral-7B-Instruct-v0.3",
    "Mixtral-8x7B-Instruct-v0.1",
    "Mixtral-8x22B-Instruct-v0.1",
    "command_r_plus",
    "llama_3_8b_instruct",
    "llama_3_70b_instruct",
    "Qwen2-7B-Instruct",
    "Qwen2-72B-Instruct",
    "gpt-3.5-turbo-0125",
    "gpt-4o-0513"
]

mmlu_models = {
    "phi-3-mini-128k-instruct": 68.1,
    "phi-3-medium-128k-instruct": 76.6,
    "Mistral-7B-Instruct-v0.1": 55.38,
    "Mistral-7B-Instruct-v0.2":  60.78,
    "Mistral-7B-Instruct-v0.3": ...,
    "Mixtral-8x7B-Instruct-v0.1": 71.16,
    "Mixtral-8x22B-Instruct-v0.1": 77.77,
    "command_r_plus": 75.73,
    "llama_3_8b_instruct": 67.07,
    "llama_3_70b_instruct": 80.06,
    "Qwen2-7B-Instruct": 70.5,
    "Qwen2-72B-Instruct": 82.3,
    "gpt-3.5-turbo-0125": 70.00,  # old version - technical report
    "gpt-4o-0513": 88.7,
}


ANALYSIS_RESULTS_DIR = "./Leaderboard/data_analysis/analysis_results"

model_results = {}

for model in models:
    with open(ANALYSIS_RESULTS_DIR+"/"+model+".json") as f:
        model_data = json.load(f)
        model_results[model] = model_data[metric]

model_results = dict(sorted(model_results.items(), key=lambda x: x[1], reverse=True))

print("Stability")
for i, (m, v) in enumerate(model_results.items()):
    print(f"{i + 1}. {m} - {v:.2f}")


fig, ax = plt.subplots(figsize=(10, 5))
xs = model_results.keys()
ys = model_results.values()


# Plot
colors = list(map(model_2_color, xs))

ax.bar(xs, ys, color=colors)
ax.set_xticklabels(xs, rotation=90)
plt.ylim((-0.1, 1))
plt.ylabel(metric)
plt.subplots_adjust(bottom=0.43)

plt.title(f"Leaderboard {metric}")

plt.show()


