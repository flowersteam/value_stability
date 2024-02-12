import subprocess
import json
import os
from itertools import chain
import numpy as np


def run_analysis(data_dir, prefix, model, assert_n_contexts=None):
    # run evaluation script
    # print(f"Path: {data_dir}/{prefix}_{model}")
    command = f"python visualization_scripts/data_analysis.py {'--assert-n-dirs ' + assert_n_contexts if assert_n_contexts else ''} results/{data_dir}/{prefix}_{model}/*"
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    if stderr:
        print("Error:", stderr.decode())

    # extract the results
    output_lines = stdout.decode().split('\n')[-3:-1]  # Adjust indices to capture the last two lines correctly

    metrics = output_lines[0].split("\t")

    if metrics != ["Mean-Level", "Rank-Order", "Ipsative"]:
        metrics = ["Mean-Level", "Rank-Order", "Ipsative"]
        values = [np.nan, np.nan, np.nan]
    else:
        values = output_lines[1].split("\t\t")


    results = dict(zip(metrics, values))

    return results

all_models = []
all_data_dirs = []

# Define the models
models = [
    "llama_2_7b",
    "llama_2_13b",
    "llama_2_70b",  # 2 gpu
    "llama_2_7b_chat",
    "llama_2_13b_chat",
    "llama_2_70b_chat",  # 2 gpu
    "Mistral-7B-v0.1",
    "Mistral-7B-Instruct-v0.1",
    "Mistral-7B-Instruct-v0.2",
    "zephyr-7b-beta",
    "Mixtral-8x7B-v0.1-4b",  # 6h
    "Mixtral-8x7B-Instruct-v0.1-4b"  # 6h
]
all_models.extend(models)
# Define the results directory
# sim conv
data_dirs_prefixes = [
    ("results_sim_conv_tolkien_simulated_human_knows_persona", "results_pvq_sim_conv_tolkien_characters"),
    ("results_sim_conv_v2_simulated_human_knows_persona", "results_pvq_sim_conv_famous_people"),
    ("results_tolkien_ultimatum_sim_conv_v2_perm", "results_ult_sim_conv_tolkien_characters"),
    ("results_regular_ultimatum_sim_conv_v2", "results_ult_sim_conv_famous_people")
]
all_data_dirs.extend(list(zip(*data_dirs_prefixes))[0])


assert_n_contexts = 5
# prefix = "results_pvq_sim_conv_famous_people"
# prefix = "results_ult_sim_conv_famous_people"


# # # sim conv - ultimatum
# data_dirs = [
#     "results_ultimatum_sim_conv_v2_perm",
# ]
# assert_n_contexts = 5
# prefix = "results_pvq_sim_conv_tolkien_characters"




data = {}
for data_dir, prefix in data_dirs_prefixes:
    print(f"EXPERIMENT: {data_dir}")
    data[data_dir] = {}

    for model in models:
        data[data_dir][model] = run_analysis(data_dir, prefix, model, assert_n_contexts=None)

        print(model.ljust(35, ' ') + " - " + str(data[data_dir][model]))


# cacheing
with open('test.json', 'w') as fp:
    json.dump(data, fp)

with open('test.json') as f:
    data = json.load(f)


# MERGE
merge = False
if merge:
    merge_dict = {
        "pvq": ["results_sim_conv_tolkien_simulated_human_knows_persona", "results_sim_conv_v2_simulated_human_knows_persona"],
        "ult": ["results_tolkien_ultimatum_sim_conv_v2_perm", "results_regular_ultimatum_sim_conv_v2"]
    }
    data_ = {}
    for k, v in merge_dict.items():
        models = list(data[v[0]].keys())
        metrics = list(list(data[v[0]].values())[0].keys())

        data_[k] = {
            model: {
                metric: np.mean([float(data[v[0]][model][metric]),float(data[v[1]][model][metric])]) for metric in metrics
            } for model in models
        }
    data = data_
    all_data_dirs = merge_dict.keys()




# DRAW PLOTS
import matplotlib.pyplot as plt

data_dirs_2_labels = {
    # "results_sim_conv_v2_perm": "tolkien",
    "results_sim_conv_tolkien_simulated_human_knows_persona": "tolkien_pvq",
    "results_tolkien_ultimatum_sim_conv_v2_perm": "tolkien_ult",
    "results_sim_conv_v2_simulated_human_knows_persona": "famous_pvq",
    "results_regular_ultimatum_sim_conv_v2": "famous_ult"
    # "results_sim_conv_v2_perm_op_only": "option order ch.",
    # "results_sim_conv_v2_perm": "topic + option order ch.",
    # "results_sim_conv_v2_perm_base_format": "topic + option order ch.",
    # "results_weather_v2": "weather change",
    # "results_weather_v2_perm_op_only": "option order ch.",
    # "results_weather_v2_perm_op": "option order + weather ch.",
}

biggest_human_change = {
    "Rank-Order": 0.57,
    "Ipsative": 0.59,
}

num_models = len(models)
num_cols = 3  # Adjust this as needed for a better layout
num_rows = num_models // num_cols + (num_models % num_cols > 0)

for metric in ["Rank-Order", "Ipsative"]:

    # Create a figure with subplots
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))  # Adjust figsize as needed
    axs = axs.flatten()  # Flatten the array of axes for easy indexing

    for i, model in enumerate(all_models):
        data_dirs_ = [d for d in all_data_dirs if model in data[d]]
        rank_order_values = [float(data[data_dir][model][metric]) for data_dir in data_dirs_]
        min_size = 0.03  # because it's invisible otherwise
        rank_order_values = [min_size if -min_size < value < min_size else value for value in rank_order_values]

        data_dir_labels = [data_dirs_2_labels.get(d,d) for d in data_dirs_]
        axs[i].bar(data_dir_labels, rank_order_values, color=['red', 'green', 'blue'])
        # axs[i].set_xlabel('Experiment')
        axs[i].set_ylabel(metric + " stability (r)")
        axs[i].set_title(f'{model}')
        axs[i].set_ylim(-0.5, 1)
        axs[i].tick_params(axis='x')

        # humans
        axs[i].axhline(y=biggest_human_change[metric], color='gray', linestyle='--')

    # Hide any unused subplots
    for j in range(i + 1, num_rows * num_cols):
        axs[j].axis('off')

    fig.suptitle(f'{metric} Stability')
    plt.tight_layout()
    plt.subplots_adjust(top=0.90, bottom=0.05, hspace=0.8)
    plt.savefig(f'visualizations/{metric}_all_models.png')
    plt.savefig(f'visualizations/{metric}_all_models.svg')
    plt.show()  # Sh
    plt.close()
