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

parser = argparse.ArgumentParser()
parser.add_argument("--no-show", action="store_true", help="Don't show plots.")
parser.add_argument("--fig-name", type=str, default="test", help="figure to plot")
parser.add_argument("--assert-n-contexts", type=int, default=-1, help="Set to <0 for no asserts")
args = parser.parse_args()

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
    "Qwen-7B",
    "Qwen-14B",
    "Qwen-72B",
    "gpt-3.5-turbo-1106",
    "gpt-3.5-turbo-0125",
]



all_ipsative_corrs_str = "All_Ipsative_corrs"

def model_2_family(model):
    model_lower = str(model).lower()
    if "llama_2" in model_lower:
        return "LLaMa-2"
    if "llama_3" in model_lower:
        return "LLaMa-3"
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
    elif "command" in model_lower:
        return "command"
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

manual_color = None


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

def plot_comparison_matrix(models, p_values_matrix, figure_savepath, title="Model Comparison"):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.matshow(p_values_matrix, cmap='gray_r')

    # Setting axes labels
    ax.set_xticks(range(len(models)))
    ax.set_yticks(range(len(models)))
    ax.set_xticklabels(models, rotation=90)
    ax.set_yticklabels(models)

    # Title and color bar
    plt.title(title)
    plt.tight_layout()

    print(f"save to: {figure_savepath}")
    plt.savefig(figure_savepath)

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


def get_all_ro_corrs_str(paired_data_dir):
    if paired_data_dir:
        return "All_Proxy_stabilities"
    else:
        return "All_Rank-Order_stabilities"

def run_analysis(eval_script_path, data_dir, assert_n_contexts=None, paired_data_dir=None, no_ips=False):
    # run evaluation script
    # command = f"python {eval_script_path} --result-json-stdout {'--assert-n-dirs ' + str(assert_n_contexts) if assert_n_contexts else ''} {data_dir}/* {f'--paired-dirs {paired_data_dir}/*' if paired_data_dir is not None else ''} {'--no-ips' if no_ips else ''}"

    if paired_data_dir is None:
        paired_str = ""
    elif any(["json" in d for d in glob.glob(paired_data_dir + "/*/*")]):
        # jsons found
        paired_str = f'--paired-dirs {paired_data_dir}/*'
    else:
        paired_str = f'--paired-dirs {paired_data_dir}/*/*'

    command = f"python {eval_script_path} --result-json-stdout {'--assert-n-dirs ' + str(assert_n_contexts) if assert_n_contexts else ''} {data_dir}/* {data_dir}/*/* {paired_str}    {'--ips' if not no_ips else ''}"
    print("Command: ", command)
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    if stderr:
        raise ValueError(stderr)

    if stderr:
        command = f"python {eval_script_path} --result-json-stdout {'--assert-n-dirs ' + str(assert_n_contexts) if assert_n_contexts else ''} {data_dir}/*/* {f'--paired-dirs {paired_data_dir}/*/*' if paired_data_dir is not None else ''} {'--no-ips' if no_ips else ''}"
        print("(old savedir detected runing Command: ", command)
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()

        if stderr:
            raise ValueError(stdout)

    # parse json outputs
    results = json.loads(stdout)

    results[all_ipsative_corrs_str] = np.array(results[all_ipsative_corrs_str])

    return results

def parse_x_labels(x):
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
        "gpt-3.5-turbo-1106": "GPT-3.5-1106",
        "gpt-3.5-turbo-0125": "GPT-3.5-0125",
    }
    if x in x_label_map:
        return x_label_map[x]
    elif x.endswith("_msgs"):
        return x.removesuffix("_msgs")
    else:
        return x

# Define the results directory
results_dir = "results"

figure_name = args.fig_name

# list of options
# figure_name = "tolk_ro_t"
# figure_name = "fam_ro_t"
# figure_name = "no_pop_ips"
# figure_name = "religion_t"
# figure_name = "don_t"
# figure_name = "bag_t"
# figure_name = "paired_tolk_ro_uni"
# figure_name = "paired_tolk_ro_ben"
# figure_name = "paired_tolk_ro_pow"
# figure_name = "paired_tolk_ro_ach"

# default params
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
human_change_xloc = 6.8

legend_loc = None

add_legend = False
legend_title = "LLM families"

title = None

left_adjust = None
paired_dir = None
y_label = None
x_label = None
FDR_test = True
fam_min_y, fam_max_y = -0.1, 0.8

seed_strings = [f"seed_{i}" for i in range(0, 9, 2)]

if figure_name == "tolk_ro_t":

    experiment_dirs = ["stability_default_params_pvq_tolkien_characters"]
    title = "(A)"

    add_legend = True
    legend_fontsize = 17

    legend_loc = (0.001, 0.99)
    metric = "Rank-Order"
    human_change_xloc = 6.8
    show_human_change = True
    rotatation_x_labels = 90

    xticks_fontsize = 15
    yticks_fontsize = 18

    min_y, max_y = -0.1, 0.8  # RO

elif figure_name == "fam_ro_t":

    experiment_dirs = ["stability_default_params_pvq_famous_people"]

    title = "(B)"

    add_legend = False
    metric = "Rank-Order"
    human_change_xloc = 6.8

    show_human_change = True
    rotatation_x_labels = 90

    xticks_fontsize = 15
    yticks_fontsize = 18

    min_y, max_y = -0.1, 0.8  # RO

elif figure_name == "religion_t":

    rotatation_x_labels = 90

    title = "(C)"

    experiment_dirs = ["stability_default_params_religion_famous_people"]

    add_legend = False

    metric = "Rank-Order"
    show_human_change = False
    legend_fontsize = 22

    xticks_fontsize = 15
    yticks_fontsize = 18

    min_y, max_y = -0.1, 0.8  # RO

elif figure_name == "don_t":

    title = "(A)"
    experiment_dirs = ["stability_default_params_tolkien_donation_tolkien_characters"]
    add_legend = True
    legend_fontsize = 16
    metric = "Rank-Order"
    human_change_xloc = 6.8
    rotatation_x_labels = 90

    xticks_fontsize = 15
    yticks_fontsize = 18

    min_y, max_y = -0.1, 0.8  # RO

elif figure_name.startswith("bag_t"):

    title = "(B)"

    experiment_dirs = ["stability_default_params_tolkien_bag_tolkien_characters"]

    add_legend = False
    metric = "Rank-Order"
    human_change_xloc = 6.8
    rotatation_x_labels = 90

    xticks_fontsize = 15
    yticks_fontsize = 18

    min_y, max_y = -0.1, 0.8  # RO

elif figure_name == "no_pop_ips":
    experiment_dirs = ["stability_default_params_pvq_permutations_msgs"]
    seed_strings = ["3_msgs/_seed"]  # ips (only n=3)

    add_legend = True
    metric = "Ipsative"
    show_human_change = True
    human_change_xloc = -1.0

    human_data_fontsize = 8
    xticks_fontsize = 17
    yticks_fontsize = 20
    legend_fontsize = 25
    rotatation_x_labels = 90
    y_label_fontsize = 25

    # legend_loc = (0.2, 0.45)
    legend_loc = (1, 1)

    interval_figsize_x = 14
    interval_figsize_y = 7

    human_data_fontsize = 14

    min_y, max_y = -0.1, 1.0  # IPS

elif figure_name.startswith("paired_tolk_ro"):

    add_legend = False

    if figure_name.endswith("uni"):
        value_to_pair = "Universalism"
        letter = "(A)"
        add_legend = True
        legend_fontsize = 15
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

    title = letter

    y_label = f"Rank-Order stability\nwith donation"

    experiment_dirs = ["stability_default_params_pvq_tolkien_characters"]
    paired_dir = "stability_default_params_tolkien_donation_tolkien_characters"

    metric = "Rank-Order"
    show_human_change = False
    rotatation_x_labels = 90

    xticks_fontsize = 15
    yticks_fontsize = 18

    left_adjust = 0.2

    if value_to_pair in ["Power", "Achievement"]:
        min_y, max_y = -0.5, 0.1
    else:
        min_y, max_y = -0.1, 0.5

elif figure_name == "tolk_ro_msgs_more":

    manual_color=family_2_color["Mixtral"]

    # Messages on Rank-Order Tolkien
    models = ["3_msgs", "7_msgs", "11_msgs", "19_msgs", "27_msgs", "35_msgs", "43_msgs"]
    experiment_dirs = ["stability_default_params_pvq_tolkien_characters_more_msgs/Mixtral-8x7B-Instruct-v0.1"]

    seed_strings = ["1_seed"]
    add_legend = False
    FDR_test = False
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
    min_y, max_y = 0.0, 0.8  # RO

    x_label = "Simulated conversation length (n)"

else:
    raise ValueError("Unknown figure name")

if y_label is None:
    y_label = metric + " stability (r)"


if args.assert_n_contexts < 0:
    args.assert_n_contexts = None
else:
    cprint(f"Asserting {args.assert_n_contexts} contexts.", "green")

data = {}
for experiment_dir in experiment_dirs:
    print(f"{experiment_dir}")
    data[experiment_dir] = {}
    for model in models:
        print(f"\t{model}")

        data[experiment_dir][model] = {}
        for seed_str in seed_strings:
            data[experiment_dir][model][seed_str] = {}

            data_dir = os.path.join(results_dir, experiment_dir, model, seed_str)

            if paired_dir:
                paired_data_dir = os.path.join(results_dir, paired_dir, model, seed_str)
            else:
                paired_data_dir = None

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
                     str(args.assert_n_contexts), str(paired_data_dir), str(no_ips)
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
                        eval_script_path=eval_script_path, data_dir=data_dir, paired_data_dir=paired_data_dir,
                        assert_n_contexts=args.assert_n_contexts,
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

            keys_to_print = ["Rank-Order", "Ipsative"]
            metrs_str = {k: np.round(v, 2) for k, v in data[experiment_dir][model][seed_str].items() if k in keys_to_print}
            print(f"\t\t- {seed_str} : {metrs_str}")


human_data_color = "black"

human_change_10_12 = {"Rank-Order": 0.569, "Ipsative": 0.66}
human_change_20_28 = {"Rank-Order": 0.657, "Ipsative": 0.59}

print(f"Metric: {metric}")


fig, axs = plt.subplots(1, 1, figsize=(interval_figsize_x, interval_figsize_y))
axs = [axs]


all_ro_corrs_str = get_all_ro_corrs_str(paired_data_dir)

for plt_i, experiment_dir in enumerate(experiment_dirs):

    if show_human_change:

        axs[plt_i].axhline(y=human_change_10_12[metric], color=human_data_color, linestyle=':', zorder=0)
        axs[plt_i].text(human_change_xloc, human_change_10_12[metric] + 0.01, "Human value stability between ages 10 and 12",
                        fontsize=human_data_fontsize, color=human_data_color)

        axs[plt_i].axhline(y=human_change_20_28[metric], color=human_data_color, linestyle=':', zorder=0)
        axs[plt_i].text(human_change_xloc, human_change_20_28[metric] + 0.01, "Human value stability between ages 20 and 28",
                        fontsize=human_data_fontsize, color=human_data_color)

    # plot bars
    plt.subplots_adjust(left=left_adjust, top=0.90, bottom=0.5, hspace=0.8)
    xs = models
    xs = [parse_x_labels(x) for x in xs]

    if figure_name.startswith("paired_tolk_ro"):
        scores = np.array([[data[experiment_dir][model][seed_str]['Proxy_stability'][value_to_pair] for seed_str in seed_strings] for model in models])
        all_scores = [
            np.array(list(itertools.chain(*[
                (itertools.chain(*data[experiment_dir][model][seed_str][all_ro_corrs_str][value_to_pair].values()) if all_ro_corrs_str in data[experiment_dir][model][seed_str] else [np.nan])
                for seed_str in seed_strings
            ]))) for model in models
        ]

    else:
        scores = np.array([[data[experiment_dir][model][seed_str][metric] for seed_str in seed_strings] for model in models])
        all_scores = [
            np.array(list(itertools.chain(*[
                (itertools.chain(*data[experiment_dir][model][seed_str][all_ro_corrs_str].values()) if all_ro_corrs_str in data[experiment_dir][model][seed_str] else [np.nan])
                for seed_str in seed_strings
            ]))) for model in models
        ]

    assert len(models) == len(scores)

    ys = scores.mean(axis=1)

    if metric == "Ipsative":
        n_msgs = models
        # [n_msgs, n_seeds, n_pairs, n_personas]
        all_corrs = np.array([[
            data[experiment_dir][n_msg][seed_str][all_ipsative_corrs_str] for seed_str in seed_strings
        ] for n_msg in n_msgs])

        all_corrs = all_corrs.mean(2)  # mean over context pairs

        # SI over what -> seed and personas
        all_corrs = all_corrs.reshape(len(n_msgs), -1)

        tick_len_se = np.array([st.sem(a) for a in all_corrs])

        scores = all_corrs

    elif metric == "Rank-Order":
        tick_len_se = np.array([st.sem(a) for a in scores])
        assert all((tick_len_se >= 0) | np.isnan(tick_len_se))

    if np.isnan(ys).all():
        raise ValueError("All models are nan.")

    print("Results")
    for x, y, t in zip(xs, ys, tick_len_se):
        print(f"{x}: {y:.5f} +/- {t:.3f}")

    if manual_color is None:
        cs = [family_2_color.get(model_2_family(x)) for x in xs]
    else:
        cs = [manual_color for x in xs]

    cs = ["black" if c is None else c for c in cs]

    labs = [model_2_family(x) for x in xs]

    axs[plt_i].bar(xs, ys, yerr=tick_len_se, color=cs, label=labs)

    assert len(experiment_dirs) == 1

    axs[plt_i].set_ylim(min_y, max_y)
    axs[plt_i].set_xticklabels([parse_x_labels(m) for m in models], rotation=rotatation_x_labels, fontsize=xticks_fontsize)
    axs[plt_i].set_yticklabels(map(lambda x: np.round(x, round_y_lab), axs[plt_i].get_yticks()), fontsize=yticks_fontsize)

    axs[plt_i].set_ylabel(y_label, fontsize=y_label_fontsize)
    if x_label is not None:
        axs[plt_i].set_xlabel(x_label, fontsize=y_label_fontsize)

    if title:
        axs[plt_i].set_title(title, fontsize=title_fontsize)

    if add_legend:
        legend_without_duplicate_labels(axs[plt_i], loc="best", title=legend_title, legend_loc=legend_loc)


plt.tight_layout()

fig_path = f'PLOSONE/data_analysis/visualizations/{figure_name}'
os.makedirs(os.path.dirname(fig_path), exist_ok=True)
savepath = f'{fig_path}.png'

print(f"save to: {savepath}")
plt.savefig(savepath)

if not args.no_show:
    plt.show()  # Sh
plt.close()
# plt.draw()


if FDR_test:
    # FDR
    p_values_corrected_matrix = FDR(scores)

    binary_matrix = (p_values_corrected_matrix < 0.05).astype(int)
    models_labels = [parse_x_labels(m) for m in models]

    # fig_path = f'visualizations/{figure_name}_comparison.pdf'
    comp_savepath = f'{fig_path}_comparison.svg'

    plot_comparison_matrix(models_labels, binary_matrix, figure_savepath=comp_savepath, title=title)
