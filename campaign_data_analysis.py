import sys
import glob
import math
import subprocess
import json
import os
import itertools
import numpy as np
import matplotlib.pyplot as plt
import hashlib
import checksumdir
import inspect
import scipy.stats as st
from termcolor import cprint
import argparse

# use this command to parse SVGs to PDFs
# for f in *; do DISPLAY= inkscape $f --export-pdf="${f%.*}.pdf"; done
# crop pdf images with
# sudo apt-get install texlive-extra-utils
# for f in *_Fig.pdf; do pdfcrop $f $f; done

parser = argparse.ArgumentParser()
parser.add_argument("--no-show", action="store_true")
parser.add_argument("--fig-name", type=str, default="test")
parser.add_argument("--assert-n-contexts", type=int, default=-1, help="Set to <0 for no asserts")
parser.add_argument("--all-models", action="store_true")
args = parser.parse_args()

if args.all_models:
    models = sorted([m.strip(".json") for m in os.listdir("./models/configs")])
else:
    models = [
        "llama_2_7b",
        "llama_2_13b",
        "llama_2_70b",
        "llama_2_7b_chat",
        "llama_2_13b_chat",
        "llama_2_70b_chat",
        "Mistral-7B-v0.1",
        "Mistral-7B-Instruct-v0.1",
        "Mistral-7B-Instruct-v0.2",
        "zephyr-7b-beta",
        "Mixtral-8x7B-v0.1-4b",
        "Mixtral-8x7B-Instruct-v0.1-4b",
        "Mixtral-8x7B-v0.1",
        "Mixtral-8x7B-Instruct-v0.1",
        "phi-1",
        "phi-2",
        "phi-3",
        "Qwen-7B",
        "Qwen-14B",
        "Qwen-72B",
        "Qwen1.5-72B-Chat",
        "gpt-3.5-turbo-1106",
        "gpt-3.5-turbo-0125",
    ]

assert len(set(models)) == len(models)


def model_2_family(model):
    model_lower = model.lower()
    if "llama_2" in model_lower:
        return "LLaMa-2"
    elif "mixtral" in model_lower:
        return "Mixtral"
    elif "mistral" in model_lower or "zephyr" in model_lower:
       return "Mistral"
    elif "phi" in model_lower:
        return "Phi"
    elif "qwen" in model_lower:
        return "Qwen"
    elif "gpt" in model_lower:
        return "GPT"
    elif "dummy" == model_lower:
        return "dummy"
    elif "random" == model_lower:
        return "random"
    else:
        return model


family_2_color = {
    "LLaMa-2": "blue",
    "Mixtral": "orange",
    "Mistral": "green",
    "Phi": "red",
    "Qwen": "purple",
    "GPT": "black",
    "dummy": "brown",
    "random": "brown"
}

family_2_linestyle = {
    "LLaMa-2": ":",
    "Mixtral": "-",
    "Mistral": "dashdot",
    "Phi":  (0, (3, 5, 1, 5, 1, 5)),
    "Qwen": "--",
    # "GPT": "-",
    # "dummy": "-"
}

def FDR(scores):
    from scipy.stats import ttest_ind
    from statsmodels.stats.multitest import multipletests

    # Compute pairwise t-tests
    n_models = scores.shape[0]
    p_values = np.ones((n_models, n_models))  # Initialize a matrix of p-values

    for i in range(n_models):
        for j in range(i + 1, n_models):  # No need to test against itself or repeat comparisons
            stat, p_value = ttest_ind(scores[i], scores[j])
            p_values[i, j] = p_value
            p_values[j, i] = p_value  # Symmetric matrix

    # Flatten the p-value matrix and remove ones to prepare for FDR correction
    p_values_flat = p_values[np.tril_indices(n_models)]
    # Apply FDR correction
    reject, p_values_corrected, _, _ = multipletests(p_values_flat, alpha=0.05, method='fdr_bh')

    # Reshape the corrected p-values back into a matrix
    p_values_corrected_matrix = np.zeros((n_models, n_models))
    p_values_corrected_matrix[np.tril_indices(n_models)] = p_values_corrected
    p_values_corrected_matrix += p_values_corrected_matrix.T  # Make symmetric

    return p_values_corrected_matrix

def plot_comparison_matrix(models, p_values_matrix, figure_name, title="Model Comparison"):
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.matshow(p_values_matrix, cmap='gray_r')

    # Setting axes labels
    ax.set_xticks(range(len(models)))
    ax.set_yticks(range(len(models)))
    ax.set_xticklabels(models, rotation=90)
    ax.set_yticklabels(models)

    # Title and color bar
    plt.title(title)
    # fig.colorbar(cax)
    plt.tight_layout()

    fig_path = f'visualizations/{figure_name}_comparison.pdf'
    print(f"save to: {fig_path}")
    plt.savefig(fig_path)

    if not args.no_show:
        plt.show()  # Sh

    plt.close()


def legend_without_duplicate_labels(ax, loc="best", title=None, legend_loc=None):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    # axs[plt_i].legend(bbox_to_anchor=legend_loc, loc="best")
    if legend_loc:
        loc="upper left"
    else:
        loc="best"

    ax.legend(*zip(*unique), loc=loc, title=title, fontsize=legend_fontsize, title_fontsize=legend_fontsize, bbox_to_anchor=legend_loc)

def get_all_ipsative_corrs_str(default_profile):

    if default_profile is None:
        return "All_Ipsative_corrs"
    else:
        return "All_Ipsative_corrs_default_profile"


def get_all_ro_corrs_str(RO_neutral, paired_data_dir):
    assert RO_neutral != paired_data_dir
    if RO_neutral:
        return "All_Neutral_Rank-Order_stabilities"
    elif paired_data_dir:
        return "All_Proxy_stabilities"
    else:
        return "All_Rank-Order_stabilities"

def run_analysis(eval_script_path, data_dir, assert_n_contexts=None, default_profile=None, paired_data_dir=None, RO_neutral=False, RO_neutral_data_dir=None, no_ips=False):
    # run evaluation script
    command = f"python {eval_script_path} --result-json-stdout {'--assert-n-dirs ' + str(assert_n_contexts) if assert_n_contexts else ''} {f'--default-profile {default_profile}' if default_profile is not None else ''} {data_dir}/* {f'--paired-dirs {paired_data_dir}/*/*' if paired_data_dir is not None else ''} {f'--neutral-ranks --neutral-dir {RO_neutral_data_dir}' if RO_neutral else ''} {'--no-ips' if no_ips else ''}"
    print("Command: ", command)
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    if stderr:
        command = f"python {eval_script_path} --result-json-stdout {'--assert-n-dirs ' + str(assert_n_contexts) if assert_n_contexts else ''} {f'--default-profile {default_profile}' if default_profile is not None else ''} {data_dir}/*/* {f'--paired-dirs {paired_data_dir}/*/*' if paired_data_dir is not None else ''} {f'--neutral-ranks --neutral-dir {RO_neutral_data_dir}' if RO_neutral else ''} {'--no-ips' if no_ips else ''}"
        print("(old savedir detected runing Command: ", command)
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()

    # parse json outputs
    results = json.loads(stdout)

    all_ipsative_corrs_str = get_all_ipsative_corrs_str(default_profile)
    results[all_ipsative_corrs_str] = np.array(results[all_ipsative_corrs_str])

    return results

all_data_dirs = []






x_label_map = {
    "dummy": "random",
    "llama_2_7b":  "LLaMa_2_7b",
    "llama_2_13b": "LLaMa_2_13b",
    "llama_2_70b": "LLaMa_2_70b",
    "llama_2_7b_chat": "LLaMa_2_7b_chat",
    "llama_2_13b_chat": "LLaMa_2_13b_chat",
    "llama_2_70b_chat": "LLaMa_2_70b_chat",
    "phi-2": "Phi-2",
    "phi-1": "Phi-1",

}
x_label_map = {**x_label_map, **{k: k.replace("_msgs", "") for k in ["1_msgs", "3_msgs", "5_msgs", "7_msgs", "9_msgs"]}}

x_label_map = {**x_label_map, **{
    "gpt-3.5-turbo-1106": "GPT-3.5-1106",
    "gpt-3.5-turbo-0125": "GPT-3.5-0125",
}}


# Define the results directory
# sim conv

add_legend = False
bars_as_plot = False
label_ = None

results_dir = "results"
# experiment_dirs = [
#     "sim_conv_pvq_tolkien_characters_seeds",
#     # "sim_conv_pvq_famous_people_seeds",
#     # "sim_conv_pvq_tolkien_characters_seeds_NO_SYSTEM",
#     # "sim_conv_tolkien_donation_tolkien_characters_seeds",
# ]
# if "permutations_msgs" in experiment_dirs[0]:
#     seed_strings = [f"{i}_msgs/_seed" for i in range(1, 10, 2)]  # msgs (show trends
#     # seed_strings = ["3_msgs/_seed"] # ips (only n=3)
# else:
#     seed_strings = [f"{i}_seed" for i in range(1, 10, 2)]
#     # seed_strings = [f"{i}_seed" for i in range(3, 10, 2)]

add_tolkien_ipsative_curve = True
bar_plots = False

metric = "Rank-Order"
# metric = "Ipsative"

ci_ticks = False

# Vérifie si au moins un argument a été passé
figure_name = args.fig_name

# list of options
# figure_name = "tolk_ro_t"
# figure_name = "fam_ro_t"
# figure_name = "no_pop_ips"
# figure_name = "no_pop_msgs"
# figure_name = "tolk_ro_msgs"
# figure_name = "religion_t"
# figure_name = "don_t"
# figure_name = "bag_t"
# figure_name = "paired_tolk_ro_uni"
# figure_name = "paired_tolk_ro_ben"
# figure_name = "paired_tolk_ro_pow"
# figure_name = "paired_tolk_ro_ach"
# app
# figure_name = "tolk_ips_msgs"
# figure_name = "tolk_ips_msgs_default_prof"
# figure_name = "tolk_ro_msgs_neutral"
# figure_name = "llama_sys_no_sys"

rotatation_x_labels = 0

legend_fontsize = 18
human_data_fontsize = 12
xticks_fontsize = 15
yticks_fontsize = 15
y_label_fontsize = 25
x_label_fontsize = 20
title_fontsize = 18

interval_figsize_x = 8
interval_figsize_y = 7

round_y_lab = 1

show_human_change = False
legend_loc = None

legend_title = "LLM families"

title=None

default_profile = None

add_tolkien_ro_curve = False
add_tolkien_ipsative_curve = False

left_adjust = None
paired_dir = None
y_label = None

RO_neutral = False

# FDR rest
FDR_test = True

# Families legend
families_plot = False
fam_min_y, fam_max_y = -0.1, 0.8

if figure_name == "no_pop_msgs":
    experiment_dirs = ["sim_conv_pvq_permutations_msgs"]
    seed_strings = [f"{i}_msgs/_seed" for i in range(1, 10, 2)]  # msgs (show trends

    FDR_test = False

    add_tolkien_ipsative_curve = True
    bar_plots = False
    models = [
        "Mixtral-8x7B-Instruct-v0.1",
        "Mixtral-8x7B-Instruct-v0.1-4b",  # 6h
        "zephyr-7b-beta",
        "Mistral-7B-Instruct-v0.2",
        "Mistral-7B-Instruct-v0.1",
        "Qwen-72B",
        "Qwen-14B",
        "Qwen-7B",
        "llama_2_70b_chat",  # 2 gpu
        "llama_2_70b",  # 2 gpu
        "phi-2",
        "gpt-3.5-turbo-0125",
    ]
    metric = "Ipsative"
    human_change_xloc = -1.0
    msgs_ro_tolk = False

    min_y, max_y = -0.1, 1.0  # IPS
    legend_fontsize = 22
    xticks_fontsize = 20
    yticks_fontsize = 20
    y_label_fontsize = 40
    x_label_fontsize = 30
    interval_figsize_x = 14
    interval_figsize_y = 7

elif figure_name == "tolk_ips_msgs_default_prof":

    # Messages on Ips Tolkien
    models = ["1_msgs", "3_msgs", "5_msgs", "7_msgs", "9_msgs"]
    experiment_dirs = ["sim_conv_pvq_tolkien_characters_msgs/Mixtral-8x7B-Instruct-v0.1"]
    seed_strings = [f"{i}_seed" for i in range(1, 10, 2)]
    y_label = "Stability"

    bar_plots = True
    bars_as_plot = True
    add_tolkien_ro_curve = True
    add_tolkien_ipsative_curve = False
    msgs_ro_tolk = True
    legend_title = None
    legend_fontsize = 14

    label_ = "Ipsative stability (with\n  the default profile)"

    metric = "Ipsative_default_profile"
    default_profile = "results/sim_conv_pvq_permutations_msgs/Mixtral-8x7B-Instruct-v0.1/9_msgs/_seed/results_sim_conv_permutations_Mixtral-8x7B-Instruct-v0.1/pvq_test_Mixtral-8x7B-Instruct-v0.1_data_pvq_pvq_auto__permutations_50_permute_options_5_no_profile_True_format_chat___2024_02_14_20_47_27"
    add_legend = True
    human_change_xloc = 6.8
    show_human_change = False

    min_y, max_y = 0.3, 0.8  # IPS
    round_y_lab = 2

    left_adjust = 0.15

    interval_figsize_x = 6
    interval_figsize_y = 6

    # PLOSONE
    interval_figsize_x = 6
    interval_figsize_y = 4

elif figure_name == "no_pop_ips":
    experiment_dirs = ["sim_conv_pvq_permutations_msgs"]
    seed_strings = ["3_msgs/_seed"]  # ips (only n=3)

    add_tolkien_ipsative_curve = False
    bar_plots = True
    add_legend = True
    metric = "Ipsative"
    human_change_xloc = -1.0
    msgs_ro_tolk = False

    show_human_change = True

    human_data_fontsize = 8
    xticks_fontsize = 17
    yticks_fontsize = 20
    legend_fontsize = 25
    rotatation_x_labels = 90
    y_label_fontsize = 30

    # legend_loc = (0.2, 0.45)
    legend_loc = (1, 1)

    interval_figsize_x = 14
    interval_figsize_y = 7

    human_data_fontsize = 14

    min_y, max_y = -0.1, 1.0  # IPS

elif figure_name.startswith("tolk_ro_t"):

    experiment_dirs = ["stability_default_params_pvq_tolkien_characters"]
    seed_strings = [f"seed_{i}" for i in range(0, 9, 2)]

    add_tolkien_ipsative_curve = False
    bar_plots = True
    add_legend = True
    legend_loc = (0.001, 0.99)
    metric = "Rank-Order"
    human_change_xloc = 6.8
    msgs_ro_tolk = False
    show_human_change = True
    legend_fontsize = 22
    rotatation_x_labels = 90

    xticks_fontsize = 15
    yticks_fontsize = 18

    min_y, max_y = -0.1, 0.8  # RO

elif figure_name.startswith("religion_t"):

    rotatation_x_labels = 90

    # title = "Religion stability of real world persons"
    # title = "(C)"

    # experiment_dirs = ["sim_conv_religion_famous_people_seeds"]
    # seed_strings = [f"{i}_seed" for i in range(1, 10, 2)]
    #
    # experiment_dirs = ["RERUN_sim_conv_religion_famous_people_seeds"]
    # seed_strings = [f"{i}_seed" for i in range(0, 9, 2)]
    # models = [
    #     "llama_2_7b",
    #     "llama_2_13b",
    #     "llama_2_70b",  # 2 gpu
    #     "llama_2_7b_chat",
    #     "llama_2_13b_chat",
    #     # "llama_2_70b_chat",  # 2 gpu
    #     "Mistral-7B-v0.1",
    #     "Mistral-7B-Instruct-v0.1",
    #     "Mistral-7B-Instruct-v0.2",
    #     "zephyr-7b-beta",
    #     "Mixtral-8x7B-v0.1-4b",  # 6h
    #     "Mixtral-8x7B-Instruct-v0.1-4b",  # 6h
    #     "Mixtral-8x7B-v0.1",
    #     "Mixtral-8x7B-Instruct-v0.1",
    #     "phi-1",
    #     "phi-2",
    #     "Qwen-7B",
    #     "Qwen-14B",
    #     "Qwen-72B",
    #     "gpt-3.5-turbo-1106",
    #     "gpt-3.5-turbo-0125",
    #
    # ]
    #
    # experiment_dirs = ["stability_religion_famous_people"]
    # seed_strings = [f"seed_{i}" for i in range(0, 9, 2)]


    experiment_dirs = ["stability_default_params_religion_famous_people"]
    seed_strings = [f"seed_{i}" for i in range(0, 9, 2)]

    add_tolkien_ipsative_curve = False
    bar_plots = True
    add_legend = False

    metric = "Rank-Order"
    msgs_ro_tolk = False
    show_human_change = False
    legend_fontsize = 22

    xticks_fontsize = 15
    yticks_fontsize = 18

    min_y, max_y = -0.1, 0.8  # RO

elif figure_name.startswith("paired_tolk_ro"):

    if figure_name.endswith("uni"):
        value_to_pair = "Universalism"
        letter = "(A)"
    elif figure_name.endswith("ben"):
        value_to_pair = "Benevolence"
        letter = "(B)"
    elif figure_name.endswith("pow"):
        value_to_pair = "Power"
        letter = "(C)"
    elif figure_name.endswith("ach"):
        value_to_pair = "Achievement"
        letter = "(D)"
    else:
        raise ValueError(f"Undefined figure name: {figure_name}")

    # title = f"{letter} {value_to_pair}"

    # y_label = f"Rank-Order stability\n{value_to_pair}-Donation"
    y_label = f"Rank-Order stability\nwith donation"

    # experiment_dirs = ["sim_conv_pvq_tolkien_characters_seeds"]
    # paired_dir = "sim_conv_tolkien_donation_tolkien_characters_seeds"
    # seed_strings = [f"{i}_seed" for i in range(1, 10, 2)]

    experiment_dirs = ["RERUN_sim_conv_pvq_tolkien_characters_seeds"]
    paired_dir = "RERUN_sim_conv_tolkien_donation_tolkien_characters_seeds"
    seed_strings = [f"{i}_seed" for i in range(0, 9, 2)]

    add_tolkien_ipsative_curve = False
    bar_plots = True

    if value_to_pair == "Universalism":
        add_legend = True
        legend_fontsize = 20
    else:
        add_legend = False

    metric = "Rank-Order"
    msgs_ro_tolk = False
    show_human_change = False
    human_change_xloc = 6.8
    rotatation_x_labels = 90

    xticks_fontsize = 15
    yticks_fontsize = 18

    left_adjust = 0.2

    if value_to_pair in ["Power", "Achievement"]:
        min_y, max_y = -0.5, 0.1
    else:
        min_y, max_y = -0.1, 0.5

elif figure_name.startswith("fam_ro_t"):

    # title = "Personal value stability of real world personas with PVQ"
    # title = "(B)"

    # experiment_dirs = ["sim_conv_pvq_famous_people_seeds"]
    # seed_strings = [f"{i}_seed" for i in range(1, 10, 2)]

    # experiment_dirs = ["RERUN_sim_conv_pvq_famous_people_seeds"]
    # seed_strings = [f"{i}_seed" for i in range(0, 9, 2)]

    experiment_dirs = ["stability_default_params_pvq_famous_people"]
    seed_strings = [f"seed_{i}" for i in range(0, 9, 2)]

    add_tolkien_ipsative_curve = False
    bar_plots = True
    add_legend = False
    metric = "Rank-Order"
    human_change_xloc = 6.8
    msgs_ro_tolk = False

    show_human_change = True
    rotatation_x_labels = 90

    xticks_fontsize = 15
    yticks_fontsize = 18

    min_y, max_y = -0.1, 0.8  # RO

elif figure_name.startswith("don_t"):

    # title = "Donation stability of fictional characters"
    # title = "(A)"
    # experiment_dirs = ["sim_conv_tolkien_donation_tolkien_characters_seeds"]
    # seed_strings = [f"{i}_seed" for i in range(1, 10, 2)]

    # experiment_dirs = ["RERUN_sim_conv_tolkien_donation_tolkien_characters_seeds"]
    # seed_strings = [f"{i}_seed" for i in range(0, 9, 2)]

    experiment_dirs = ["stability_default_params_tolkien_donation_tolkien_characters"]
    seed_strings = [f"seed_{i}" for i in range(0, 9, 2)]

    # models = [
    #     "llama_2_7b",
    #     "llama_2_13b",
    #     "llama_2_70b",  # pushed on GPU
    #     "llama_2_7b_chat",
    #     "llama_2_13b_chat",
    #     "llama_2_70b_chat",  # pushed on GPU
    #     "Mistral-7B-v0.1",
    #     "Mistral-7B-Instruct-v0.1",
    #     "Mistral-7B-Instruct-v0.2",
    #     "zephyr-7b-beta",
    #     "Mixtral-8x7B-v0.1-4b",  # 6h  # not on GPU still fail??
    #     "Mixtral-8x7B-Instruct-v0.1-4b",  # 6h # not on GPU stil fail??
    #     "Mixtral-8x7B-v0.1", # 3 good  #pushed on GPU
    #     "Mixtral-8x7B-Instruct-v0.1",# pushed on GPU
    #     "phi-1",
    #     "phi-2",
    #     "Qwen-7B",
    #     "Qwen-14B",
    #     "Qwen-72B",  # nije uspio nista izgenerirat, ali svi load (znaci zapne na prvoj generaciji?) pushed on GPU
    #     "gpt-3.5-turbo-1106",
    #     "gpt-3.5-turbo-0125",
    # ]

    add_tolkien_ipsative_curve = False
    bar_plots = True
    metric = "Rank-Order"
    human_change_xloc = 6.8
    msgs_ro_tolk = False
    rotatation_x_labels = 90

    xticks_fontsize = 15
    yticks_fontsize = 18

    min_y, max_y = -0.1, 0.8  # RO

elif figure_name.startswith("bag_t"):

    # title = "Stealing stability of fictional characters"
    # title = "(B)"

    # experiment_dirs = ["sim_conv_tolkien_bag_tolkien_characters_seeds"]
    # seed_strings = [f"{i}_seed" for i in range(1, 10, 2)]

    experiment_dirs = ["RERUN_sim_conv_tolkien_bag_tolkien_characters_seeds"]
    seed_strings = [f"{i}_seed" for i in range(0, 9, 2)]

    add_tolkien_ipsative_curve = False
    bar_plots = True
    add_legend = True
    metric = "Rank-Order"
    human_change_xloc = 6.8
    msgs_ro_tolk = False
    rotatation_x_labels = 90

    xticks_fontsize = 15
    yticks_fontsize = 18

    min_y, max_y = -0.1, 0.8  # RO

elif figure_name == "tolk_ro_msgs":
    # Messages on Rank-Order Tolkien
    models = ["1_msgs", "3_msgs", "5_msgs", "7_msgs", "9_msgs"]
    experiment_dirs = ["sim_conv_pvq_tolkien_characters_msgs/Mixtral-8x7B-Instruct-v0.1"]
    seed_strings = [f"{i}_seed" for i in range(1, 10, 2)]

    bar_plots = True
    bars_as_plot = False
    add_tolkien_ipsative_curve = False
    msgs_ro_tolk = True

    metric = "Rank-Order"
    human_change_xloc = 6.8
    interval_figsize_x = 14
    interval_figsize_y = 7

    xticks_fontsize = 25
    yticks_fontsize = 25
    y_label_fontsize = 35
    x_label_fontsize = 30

    round_y_lab = 2
    min_y, max_y = 0.25, 0.5  # RO

elif figure_name == "tolk_ro_msgs_neutral":
    # Messages on Rank-Order Tolkien
    models = ["1_msgs", "3_msgs", "5_msgs", "7_msgs", "9_msgs"]
    experiment_dirs = ["sim_conv_pvq_tolkien_characters_msgs/Mixtral-8x7B-Instruct-v0.1"]
    seed_strings = [f"{i}_seed" for i in range(1, 10, 2)]

    RO_neutral_dir = "sim_conv_pvq_tolkien_characters_seeds/Mixtral-8x7B-Instruct-v0.1"
    RO_neutral = True

    bar_plots = True

    bars_as_plot = True
    add_tolkien_ipsative_curve = False
    add_tolkien_ro_curve = True
    msgs_ro_tolk = True

    add_legend = True
    legend_title=None
    label_ = "Rank-Order stability\n  (with the neutral order)"

    metric = "Rank-Order"
    human_change_xloc = 6.8
    interval_figsize_x = 14
    interval_figsize_y = 7

    xticks_fontsize = 25
    yticks_fontsize = 25
    y_label_fontsize = 35
    x_label_fontsize = 30

    round_y_lab = 2
    min_y, max_y = 0.30, 0.58  # RO

elif figure_name == "tolk_ips_msgs":
    # Messages on Ips Tolkien
    models = ["1_msgs", "3_msgs", "5_msgs", "7_msgs", "9_msgs"]
    experiment_dirs = ["sim_conv_pvq_tolkien_characters_msgs/Mixtral-8x7B-Instruct-v0.1"]
    seed_strings = [f"{i}_seed" for i in range(1, 10, 2)]

    bar_plots = True
    bars_as_plot = True
    add_tolkien_ipsative_curve = False
    msgs_ro_tolk = True

    metric = "Ipsative"
    human_change_xloc = 6.8

    min_y, max_y = -0.1, 1  # IPS

elif figure_name == "llama_sys_no_sys":
    families_plot = False
    # title = "Personal value stability of fictional characters with PVQ"

    experiment_dirs = [
        # "sim_conv_pvq_tolkien_characters_seeds",
        # "sim_conv_pvq_tolkien_characters_seeds_NO_SYSTEM",
        ""
    ]

    seed_strings = [f"{i}_seed" for i in range(1, 10, 2)]
    models = [
        "sim_conv_pvq_tolkien_characters_seeds/llama_2_7b_chat",
        "sim_conv_pvq_tolkien_characters_seeds/llama_2_13b_chat",
        "sim_conv_pvq_tolkien_characters_seeds/llama_2_70b_chat",  # 2 gpu
        "sim_conv_pvq_tolkien_characters_seeds_NO_SYSTEM/llama_2_7b_chat",
        "sim_conv_pvq_tolkien_characters_seeds_NO_SYSTEM/llama_2_13b_chat",
        "sim_conv_pvq_tolkien_characters_seeds_NO_SYSTEM/llama_2_70b_chat",  # 2 gpu
    ]
    x_label_map = {
        "sim_conv_pvq_tolkien_characters_seeds/llama_2_7b_chat": "llama_2_7b_chat_sys",
        "sim_conv_pvq_tolkien_characters_seeds/llama_2_13b_chat": "llama_2_13b_chat_sys",
        "sim_conv_pvq_tolkien_characters_seeds/llama_2_70b_chat": "llama_2_70b_chat_sys",  # 2 gpu
        "sim_conv_pvq_tolkien_characters_seeds_NO_SYSTEM/llama_2_7b_chat": "llama_2_7b_chat_no_sys",
        "sim_conv_pvq_tolkien_characters_seeds_NO_SYSTEM/llama_2_13b_chat": "llama_2_13b_chat_no_sys",
        "sim_conv_pvq_tolkien_characters_seeds_NO_SYSTEM/llama_2_70b_chat": "llama_2_70b_chat_no_sys",  # 2 gpu
    }

    add_tolkien_ipsative_curve = False
    bar_plots = True
    add_legend = False
    metric = "Rank-Order"
    human_change_xloc = -0.5
    msgs_ro_tolk = False
    show_human_change = True
    legend_fontsize = 22
    rotatation_x_labels = 90
    show_human_changea = False

    xticks_fontsize = 15
    yticks_fontsize = 18

    min_y, max_y = -0.1, 0.8  # RO

else:
    raise ValueError("Unknown figure name")
    # scratch
    # results_dir = "results"
    # experiment_dirs = ["Temp_GS_religion_famous_people_seeds"]
    # models = ["dummy"]
    # seed_strings = ["temp_0.4", "temp_0.7", "temp_1.0", "temp_1.5"]
    #
    # add_tolkien_ipsative_curve = False
    # bar_plots = True
    # add_legend = False
    #
    # metric = "Rank-Order"
    # msgs_ro_tolk = False
    # show_human_change = False
    # legend_fontsize = 22
    #
    # xticks_fontsize = 15
    # yticks_fontsize = 18
    #
    # min_y, max_y = -0.1, 0.8  # RO

    rotatation_x_labels = 90

    # models = ["dummy"]
    experiment_dirs = ["stability_default_params_religion_famous_people"]
    seed_strings = [f"seed_{i}" for i in range(0, 9, 2)]

    add_tolkien_ipsative_curve = False
    bar_plots = True
    add_legend = False

    metric = "Rank-Order"
    msgs_ro_tolk = False
    show_human_change = False
    legend_fontsize = 22

    xticks_fontsize = 15
    yticks_fontsize = 18

    min_y, max_y = -0.1, 0.8  # RO

if y_label is None:
    y_label = metric + " stability (r)"

if add_tolkien_ipsative_curve:
    with open("tolkien_ipsative_curve_cache.json", "r") as f:
        tolkien_ipsative_curve = json.load(f)

if add_tolkien_ro_curve:
    with open("tolkien_ro_curve_cache.json", "r") as f:
        tolkien_ro_curve = json.load(f)


# confidence = 0.95

n_comp = math.comb(len(models), 2)  # n comparisons

print("N_comp:", n_comp)

confidence = 0.95


if args.assert_n_contexts < 0:
    args.assert_n_contexts = None
else:
    cprint(f"Asserting {args.assert_n_contexts} contexts.", "green")

# prefix = "results_pvq_sim_conv_famous_people"
# prefix = "results_ult_sim_conv_famous_people"


data = {}
for experiment_dir in experiment_dirs:
    print(f"{experiment_dir}")
    data[experiment_dir] = {}
    for model in models:
        print(f"\t{model}")


        data[experiment_dir][model] = {}
        for seed_str in seed_strings:
            data[experiment_dir][model][seed_str] = {}

            # data_dir = os.path.join("results", experiment_dir, model, seed_str)
            data_dir = os.path.join(results_dir, experiment_dir, model, seed_str)

            if paired_dir:
                paired_data_dir = os.path.join("results", paired_dir, model, seed_str)
            else:
                paired_data_dir = None

            if RO_neutral:
                RO_neutral_data_dir = os.path.join("results", RO_neutral_dir, seed_str)
            else:
                RO_neutral_data_dir = None

            if len(glob.glob(data_dir+"/*/*.json")) < 3 and len(glob.glob(data_dir + "/*/*/*.json")) < 3:
                print(f"No evaluation found at {data_dir}.")
                # no evaluations
                eval_data = dict(zip(["Mean-Level", "Rank-Order", "Ipsative"], [np.nan, np.nan, np.nan]))

            else:
                no_ips = metric != "Ipsative"
                # compute hash
                eval_script_path = "./visualization_scripts/data_analysis.py"
                with open(eval_script_path, 'rb') as file_obj: eval_script = str(file_obj.read())
                hash = hashlib.sha256("-".join(
                    [eval_script, inspect.getsource(run_analysis), checksumdir.dirhash(data_dir),
                     str(args.assert_n_contexts), str(False),
                     str(default_profile), str(paired_data_dir),
                     str(RO_neutral), str(RO_neutral_data_dir),
                     str(no_ips)
                     ]).encode()).hexdigest()
                cache_path = f".cache/{hash}.json"

                # check for cache
                if os.path.isfile(cache_path):
                    with open(cache_path) as f:
                        print("\t\tLoading from cache")
                        eval_data = json.load(f)

                else:
                    print("\t\tEvaluating")
                    eval_data = run_analysis(
                        eval_script_path=eval_script_path, data_dir=data_dir, assert_n_contexts=args.assert_n_contexts,
                        default_profile=default_profile,
                        paired_data_dir=paired_data_dir, RO_neutral=RO_neutral, RO_neutral_data_dir=RO_neutral_data_dir,
                        no_ips=no_ips,
                    )

                with open(cache_path, 'w') as fp:

                    class NumpyEncoder(json.JSONEncoder):
                        def default(self, obj):
                            if isinstance(obj, np.ndarray):
                                return obj.tolist()
                            return json.JSONEncoder.default(self, obj)

                    json.dump(eval_data, fp, cls=NumpyEncoder)

            data[experiment_dir][model][seed_str] = eval_data.copy()

            keys_to_print = ["Mean-Level", "Rank-Order", "Ipsative"]
            metrs_str = {k: np.round(v, 2) for k, v in data[experiment_dir][model][seed_str].items() if k in keys_to_print}
            print(f"\t\t- {seed_str} : {metrs_str}")


human_data_color = "black"

# # ips
# legend_fontsize = 8
# human_data_fontsize = 5.5

human_change_10_12 = {
    "Mean-Level": None,
    "Rank-Order": 0.569,
    "Ipsative": 0.66
}

human_change_20_28 = {
    "Mean-Level": 0.11,
    "Rank-Order": 0.657,
    "Ipsative": 0.59,
}

human_change_20_24 = {
    "Mean-Level": 0.19,
    "Rank-Order": 0.69,
    "Ipsative": 0.59,
}

human_change_24_28 = {
    "Mean-Level": 0.11,
    "Rank-Order": 0.77,
    "Ipsative": 0.65,
}

num_plots = len(experiment_dirs)
num_cols = min(len(experiment_dirs), 3)  # Adjust this as needed for a better layout
num_rows = num_plots // num_cols + (num_plots % num_cols > 0)

print(f"Metric: {metric}")


fig, axs = plt.subplots(num_rows, num_cols, figsize=(interval_figsize_x * num_cols, interval_figsize_y * num_rows))

if num_cols == 1:
    axs=[axs]
else:
    axs = axs.flatten()

all_ipsative_corrs_str = get_all_ipsative_corrs_str(default_profile)
all_ro_corrs_str = get_all_ro_corrs_str(RO_neutral, paired_data_dir)

from collections import defaultdict
family_data = defaultdict(list)

for plt_i, experiment_dir in enumerate(experiment_dirs):

    if show_human_change:

        if default_profile:
            metric_human = "Ipsative"
        else:
            metric_human = metric

        axs[plt_i].axhline(y=human_change_10_12[metric_human], color=human_data_color, linestyle=':', zorder=0)
        axs[plt_i].text(human_change_xloc, human_change_10_12[metric_human] + 0.01, "Human value stability between ages 10 and 12",
                        fontsize=human_data_fontsize, color=human_data_color)

        axs[plt_i].axhline(y=human_change_20_28[metric_human], color=human_data_color, linestyle=':', zorder=0)
        axs[plt_i].text(human_change_xloc, human_change_20_28[metric_human] + 0.01, "Human value stability between ages 20 and 28",
                        fontsize=human_data_fontsize, color=human_data_color)


    if bar_plots:

        plt.subplots_adjust(left=left_adjust, top=0.90, bottom=0.5, hspace=0.8)
        xs = models
        xs = [x_label_map.get(x, x) for x in xs]

        if figure_name.startswith("paired_tolk_ro"):
            scores = np.array([[data[experiment_dir][model][seed_str]['Proxy_stability'][value_to_pair] for seed_str in seed_strings] for model in models])

        elif RO_neutral:
            assert metric == "Rank-Order"
            scores = np.array([[data[experiment_dir][model][seed_str]["Neutral_Rank-Order"] for seed_str in seed_strings] for model in models])
        else:
            scores = np.array([[data[experiment_dir][model][seed_str][metric] for seed_str in seed_strings] for model in models])

        if figure_name.startswith("paired_tolk_ro"):
            # reorganize this
            all_scores = [
                np.array(list(itertools.chain(*[
                    (itertools.chain(*data[experiment_dir][model][seed_str][all_ro_corrs_str][value_to_pair].values()) if all_ro_corrs_str in data[experiment_dir][model][seed_str] else [np.nan])
                    for seed_str in seed_strings
                ]))) for model in models
            ]

        else:
            # reorganize this
            all_scores = [
                np.array(list(itertools.chain(*[
                    (itertools.chain(*data[experiment_dir][model][seed_str][all_ro_corrs_str].values()) if all_ro_corrs_str in data[experiment_dir][model][seed_str] else [np.nan])
                    for seed_str in seed_strings
                ]))) for model in models
            ]

        assert len(models) == len(scores)

        for model, m_scores in zip(models, scores):
            family_data[model_2_family(model)].append(m_scores)

        ys = scores.mean(axis=1)

        # get the right side of the CI
        if "sim_conv_pvq_permutations_msgs" in experiment_dir:
            assert metric == "Ipsative"
            assert len(seed_strings) == 1  # you should use plots, not bars
            # [n_modelx, cont_pairs, pop_size]
            all_corrs = np.array([data[experiment_dir][model][seed_strings[0]][all_ipsative_corrs_str] for model in models])
            all_corrs = all_corrs.mean(1)  # mean over pairs

            c2 = np.array([st.t.interval(confidence, len(a) - 1, loc=np.mean(a), scale=st.sem(a))[1] for a in all_corrs])
            tick_len_ci = c2 - ys  # half the conf interval

            tick_len_se = np.array(list(st.sem(a) for a in all_corrs))

            scores = all_corrs

        elif metric == "Ipsative":
            n_msgs = models
            # [n_msgs, n_seeds, n_pairs, n_personas]
            all_corrs = np.array([[
                data[experiment_dir][n_msg][seed_str][all_ipsative_corrs_str] for seed_str in seed_strings
            ] for n_msg in n_msgs])

            all_corrs = all_corrs.mean(2)  # mean over context pairs

            # SI over what -> seed and personas
            all_corrs = all_corrs.reshape(len(n_msgs), -1)

            # SI over what -> personas
            # SI over what -> seeds
            c2 = np.array([st.t.interval(confidence, len(a) - 1, loc=np.mean(a), scale=st.sem(a))[1] for a in all_corrs])
            tick_len_ci = c2 - ys  # half the conf interval
            tick_len_ci = None

            tick_len_se = np.array([st.sem(a) for a in all_corrs])

            scores = all_corrs

        else:
            assert metric == "Rank-Order"

            # 5 seeds = 5 samples
            c2 = np.array([st.t.interval(confidence, len(a) - 1, loc=np.mean(a), scale=st.sem(a))[1] for a in scores])

            # 5 seeds x 10 values x (5 ch 2) = 500 samples
            # assert np.allclose(np.array(all_scores).mean(1), ys)
            # c2 = np.array([st.t.interval(confidence, len(a) - 1, loc=np.mean(a), scale=st.sem(a))[1] for a in all_scores])
            tick_len_ci = c2 - ys  # half the conf interval

            print("Error bars")
            tick_len_se = np.array([st.sem(a) for a in scores])

            assert all((tick_len_se >= 0) | np.isnan(tick_len_se))

        if np.isnan(ys).all():
            raise ValueError("All models are nan.")

        print("Results")
        for x, y, t in zip(xs, ys, tick_len_se):
            print(f"{x}: {y:.5f} +/- {t:.3f}")

        if bars_as_plot:
            # used for msgs
            axs[plt_i].plot(xs, ys, label=label_)
            axs[plt_i].fill_between(xs, ys - tick_len_se, ys + tick_len_se, alpha=0.3)

            if metric == "Ipsative" and figure_name == "tolk_ips_msgs":
                tolkien_ipsative_curve = {
                    "xs": list(xs),
                    "ys": list(ys),
                    "tick_len": list(tick_len_se),
                }

                cprint("SAVING Ipsative Tolkien Mixtral-Instruct stability to CACHE", "red")
                with open("tolkien_ipsative_curve_cache.json", "w") as f:
                    json.dump(tolkien_ipsative_curve, f)

            # load ro
            if add_tolkien_ro_curve:
                cprint("Loading Rank-order Tolkien Mixtral-Instruct stability from CACHE", "red")

                xs = np.array(tolkien_ro_curve["xs"])
                xs = [x_label_map.get(x, x) for x in xs]
                ys = np.array(tolkien_ro_curve["ys"])
                shade_len = np.array(tolkien_ro_curve["tick_len"])

                lab_ = "Rank-Order stability\n  (between contexts)"
                col_ = "black"
                axs[plt_i].plot(xs, ys, label=lab_, color=col_)
                axs[plt_i].fill_between(xs, ys - shade_len, ys + shade_len, alpha=0.3, color=col_)

            if add_tolkien_ipsative_curve:
                cprint("Loading Ipsative Tolkien Mixtral-Instruct stability from CACHE", "red")

                xs = np.array(tolkien_ipsative_curve["xs"])
                xs = [x_label_map.get(x, x) for x in xs]
                ys = np.array(tolkien_ipsative_curve["ys"])
                shade_len = np.array(tolkien_ipsative_curve["tick_len"])

                lab_ = "Ipsative stability (between contexts)"
                col_ = "brown"
                axs[plt_i].plot(xs, ys, label=lab_, color=col_, zorder=0)
                axs[plt_i].fill_between(xs, ys - shade_len, ys + shade_len, alpha=0.3, color=col_, zorder=0)

        else:
            if msgs_ro_tolk:
                axs[plt_i].bar(xs, ys, yerr=tick_len_se)

                if metric == "Rank-Order" and figure_name == "tolk_ro_msgs":
                    tolkien_ro_curve = {
                        "xs": list(xs),
                        "ys": list(ys),
                        "tick_len": list(tick_len_se),
                    }

                    cprint("SAVING Rank-order Tolkien Mixtral-Instruct stability to CACHE", "red")
                    with open("tolkien_ro_curve_cache.json", "w") as f:
                        json.dump(tolkien_ro_curve, f)

            else:
                cs = [family_2_color.get(model_2_family(x), "black") for x in xs]
                labs = [model_2_family(x) for x in xs]
                axs[plt_i].bar(xs, ys, yerr=tick_len_se, color=cs, label=labs)
                if ci_ticks:
                    # axs[plt_i].bar(xs, ys, yerr=tick_len_ci, color=cs, label=labs)
                    axs[plt_i].scatter(xs, ys+tick_len_ci, marker="x", color="black", s=20, lw=0.8)
                    axs[plt_i].scatter(xs, ys-tick_len_ci, marker="x", color="black", s=20, lw=0.8)

                assert len(experiment_dirs) == 1

        axs[plt_i].set_ylim(min_y, max_y)
        axs[plt_i].set_xticklabels([x_label_map.get(m, m) for m in models], rotation=rotatation_x_labels, fontsize=xticks_fontsize)
        axs[plt_i].set_xticklabels(models, rotation=rotatation_x_labels, fontsize=xticks_fontsize)
        axs[plt_i].set_yticklabels(map(lambda x: np.round(x, round_y_lab), axs[plt_i].get_yticks()), fontsize=yticks_fontsize)

        axs[plt_i].set_ylabel(y_label, fontsize=y_label_fontsize)

        if title:
            axs[plt_i].set_title(title, fontsize=title_fontsize)

        if msgs_ro_tolk:
            axs[plt_i].set_ylim(min_y, max_y)
            axs[plt_i].set_xlabel("Simulated conversation length (n)", fontsize=x_label_fontsize)

        if add_legend:
            legend_without_duplicate_labels(axs[plt_i], loc="best", title=legend_title, legend_loc=legend_loc)

    else:

        if add_tolkien_ipsative_curve:
            cprint("Loading Ipsative Rank-order Tolkien Mixtral-Instruct stability from CACHE", "red")

            assert metric == "Ipsative"
            assert not bars_as_plot
            xs = np.array(tolkien_ipsative_curve["xs"])
            xs = [x_label_map.get(x,x) for x in xs]
            ys = np.array(tolkien_ipsative_curve["ys"])
            shade_len = np.array(tolkien_ipsative_curve["tick_len"])

            m_ = "Mixtral-8x7B-Instruct-v0.1"
            lab_ = x_label_map.get(m_, m_)+" (fict. char.)"
            axs[plt_i].plot(xs, ys, label=lab_, linestyle="-.", color="black")
            axs[plt_i].fill_between(xs, ys - shade_len, ys + shade_len, alpha=0.3, color="black")

        for model in models:
            xs = seed_strings
            ys = [data[experiment_dir][model][msg][metric] for msg in seed_strings]

            # [n_msgs, context pairs, n_pop]
            all_corrs = np.array([data[experiment_dir][model][msg][all_ipsative_corrs_str] for msg in seed_strings])
            all_corrs = np.mean(all_corrs, axis=1)  # average over context pairs
            c2 = np.array([
                st.t.interval(confidence, len(msg_corrs) - 1, loc=np.mean(msg_corrs), scale=st.sem(msg_corrs))[1] for msg_corrs in all_corrs
            ])
            shade_len = c2 - ys  # half the conf interval

            print(f"{model}: {ys}")

            family = model_2_family(model)
            # linestyle = family_2_linestyle[family]
            linestyle = "-"

            xs = [x.replace("/_seed", "") for x in xs]
            xs = [x_label_map.get(x, x) for x in xs]

            # ugly patch to make colors nicer
            c = "black" if "gpt" in model else None

            line, = axs[plt_i].plot(xs, ys, label=x_label_map.get(model, model), linestyle=linestyle, color=c)
            line_color = line.get_color()
            axs[plt_i].fill_between(xs, ys - shade_len, ys + shade_len, alpha=0.3, color=line_color)

        max_y = 0.8 if metric == "Rank-Order" else 1.0
        axs[plt_i].set_ylim(-0.1, max_y)
        axs[plt_i].set_xlim(0, len(seed_strings) - 1)
        axs[plt_i].set_ylabel(y_label, fontsize=y_label_fontsize)
        axs[plt_i].set_xlabel("Simulated conversation length (n)", fontsize=x_label_fontsize)
        # axs[plt_i].set_title(experiment_dir.replace("sim_conv_", "").replace("_seeds", ""))
        if add_legend:
            axs[plt_i].legend(bbox_to_anchor=(1.04, 1), loc="best", fontsize=legend_fontsize)

        axs[plt_i].set_xticklabels(axs[plt_i].get_xticks(), rotation=rotatation_x_labels, fontsize=xticks_fontsize)
        axs[plt_i].set_yticklabels(map(lambda  x: np.round(x, round_y_lab), axs[plt_i].get_yticks()), fontsize=yticks_fontsize)

        plt.subplots_adjust(left=0.1, top=0.95, bottom=0.2, hspace=0.8)

# Hide any unused subplots
for j in range(plt_i + 1, num_rows * num_cols):
    axs[j].axis('off')

plt.tight_layout()

# fig_path = f'visualizations/{figure_name}.svg'
fig_path = f'visualizations/{figure_name}.pdf'
print(f"save to: {fig_path}")
plt.savefig(fig_path)

if not args.no_show:
    plt.show()  # Sh
plt.close()
# plt.draw()


if FDR_test:
    # FDR
    p_values_corrected_matrix = FDR(scores)

    binary_matrix = (p_values_corrected_matrix < 0.05).astype(int)
    models_labels = [x_label_map.get(m, m) for m in models]
    plot_comparison_matrix(models_labels, binary_matrix, figure_name, title=title)

if families_plot:
    # fam_conf = 0.05

    n_comp = math.comb(len(family_data), 2)

    # sidak
    # fam_confidence = (1-0.05)**(1/n_comp)
    # bonf
    fam_confidence = 1-0.05/n_comp

    families = list(family_data.keys())
    family_scores = np.array([np.array(family_data[f]).mean(axis=0) for f in families])  # n seeds per family
    family_means = family_scores.mean(axis=1)
    family_CIs = np.array([st.t.interval(fam_confidence, len(a) - 1, loc=np.mean(a), scale=st.sem(a))[1] for a in family_scores])
    family_tick_len_ci = family_CIs - family_means  # half the conf interval
    colors = [family_2_colorget(f, "black") for f in families]
    family_tick_len_se = np.array([st.sem(a) for a in family_scores])

    tick_len_se = np.array([st.sem(a) for a in family_scores])

    plt.gca().set_title(title)
    plt.bar(families, family_means, yerr=family_tick_len_se, color=colors, label=families)

    if ci_ticks:
        plt.scatter(families, family_means + family_tick_len_ci, marker="x", color="black", s=20, lw=0.8)
        plt.scatter(families, family_means - family_tick_len_ci, marker="x", color="black", s=20, lw=0.8)

    plt.gca().set_xticklabels(families, rotation=rotatation_x_labels, fontsize=xticks_fontsize)
    plt.gca().set_yticklabels(map(lambda x: np.round(x, round_y_lab), axs[plt_i].get_yticks()), fontsize=yticks_fontsize)
    plt.gca().set_ylabel(y_label, fontsize=y_label_fontsize)

    plt.tight_layout()

    plt.gca().set_ylim(fam_min_y, fam_max_y)

    fig_path = f'visualizations/{figure_name}_families.svg'
    print(f"save to: {fig_path}")
    plt.savefig(fig_path)

    if not args.no_show:
        plt.show()  # Sh

    plt.close()

