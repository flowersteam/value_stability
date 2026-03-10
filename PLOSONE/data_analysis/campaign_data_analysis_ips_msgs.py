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
        "llama_3_8b_instruct",
        "llama_3_70b_instruct",
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


assert len(set(models)) == len(models)


def model_2_family(model):
    model_lower = model.lower()
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
    "LLaMa-3": "cornflowerblue",
    "Mixtral": "orange",
    "Mistral": "green",
    "Phi": "red",
    "Qwen": "purple",
    "GPT": "black",
    "command": "gold",
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


def legend_without_duplicate_labels(ax, loc="best", title=None, legend_loc=None):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    # axs[plt_i].legend(bbox_to_anchor=legend_loc, loc="best")
    if legend_loc:
        loc="upper left"
    else:
        loc="best"

    ax.legend(*zip(*unique), loc=loc, title=title, fontsize=legend_fontsize, title_fontsize=legend_fontsize, bbox_to_anchor=legend_loc)

def get_all_ipsative_corrs_str():
    return "All_Ipsative_corrs"


def run_analysis(eval_script_path, data_dir, assert_n_contexts=None):
    # run evaluation script
    command = f"python {eval_script_path} --result-json-stdout --ips {'--assert-n-dirs ' + str(assert_n_contexts) if assert_n_contexts else ''} {data_dir}/*"
    print("Command: ", command)
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    if stderr:
        command = f"python {eval_script_path} --result-json-stdout --ips {'--assert-n-dirs ' + str(assert_n_contexts) if assert_n_contexts else ''} {data_dir}/*/*"
        print("(old savedir detected runing Command: ", command)
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()

    # parse json outputs
    results = json.loads(stdout)

    all_ipsative_corrs_str = get_all_ipsative_corrs_str()
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
x_label_map = {**x_label_map, **{k: k.replace("_msgs", "") for k in ["1_msgs", "3_msgs", "5_msgs", "7_msgs", "9_msgs", "43_msgs"]}}

x_label_map = {**x_label_map, **{
    "gpt-3.5-turbo-1106": "GPT-3.5-1106",
    "gpt-3.5-turbo-0125": "GPT-3.5-0125",
}}


# Define the results directory
# sim conv

add_legend = False
bars_as_plot = False
label_ = None

results_dir = "PLOSONE/results"

add_tolkien_ipsative_curve = True
bar_plots = False


figure_name = args.fig_name


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

legend_loc = None

legend_title = "LLM families"

title = None

add_tolkien_ipsative_curve = False

left_adjust = None
y_label = None

x_label = None


if figure_name == "ips_msgs":
    experiment_dirs = ["stability_default_params_pvq_permutations_msgs"]
    seed_strings = [f"{i}_msgs/_seed" for i in range(1, 10, 2)] + ["43_msgs/_seed"]  # msgs (show trends)

    add_tolkien_ipsative_curve = True
    bar_plots = False
    models = [
        "Mixtral-8x7B-Instruct-v0.1",
        "Mixtral-8x7B-Instruct-v0.1-4b",  # 6h
        "gpt-3.5-turbo-0125",
        "zephyr-7b-beta",
        "Mistral-7B-Instruct-v0.2",
        "Mistral-7B-Instruct-v0.1",
        "Qwen-72B",
        "Qwen-14B",
        "Qwen-7B",
        "llama_2_70b_chat",  # 2 gpu
        "llama_2_70b",  # 2 gpu
        "phi-2",
        "phi-1",
    ]
    metric = "Ipsative"

    add_legend = True

    min_y, max_y = -0.1, 1.0  # IPS

    legend_fontsize = 22

    yticks_fontsize = 27
    xticks_fontsize = 27

    y_label_fontsize = 40
    # x_label_fontsize = 20
    x_label_fontsize = 35

    # interval_figsize_x = 14
    interval_figsize_x = 21
    interval_figsize_y = 7

    x_label = "Simulated conversation length (n)"

elif figure_name == "tolk_ips_msgs":
    # Messages on Ips Tolkien
    models = ["1_msgs", "3_msgs", "5_msgs", "7_msgs", "9_msgs", "43_msgs"]
    experiment_dirs = ["sim_conv_pvq_tolkien_characters_msgs/Mixtral-8x7B-Instruct-v0.1"]
    seed_strings = ["1_seed"]

    bar_plots = True
    bars_as_plot = True
    add_tolkien_ipsative_curve = False

    metric = "Ipsative"
    human_change_xloc = 6.8

    min_y, max_y = -0.1, 1  # IPS

else:
    raise ValueError("Unknown figure name")

if y_label is None:
    y_label = metric + " stability (r)"

if add_tolkien_ipsative_curve:
    with open("PLOSONE/tolkien_ipsative_curve_cache.json", "r") as f:
        tolkien_ipsative_curve = json.load(f)


n_comp = math.comb(len(models), 2)  # n comparisons

print("N_comp:", n_comp)


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

            n_evals_found = max(len(glob.glob(data_dir+"/*/*.json")), len(glob.glob(data_dir + "/*/*/*.json")))

            if n_evals_found < 3:
                print(f"{n_evals_found} < 3  evaluations found at {data_dir}.")
                # no evaluations
                eval_data = dict(zip(["Mean-Level", "Rank-Order", "Ipsative"], [np.nan, np.nan, np.nan]))

            else:
                # compute hash
                eval_script_path = "./visualization_scripts/data_analysis.py"
                with open(eval_script_path, 'rb') as file_obj: eval_script = str(file_obj.read())
                hash = hashlib.sha256("-".join([
                    eval_script,
                    inspect.getsource(run_analysis),
                    checksumdir.dirhash(data_dir),
                    str(args.assert_n_contexts)
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
                        eval_script_path=eval_script_path, data_dir=data_dir, assert_n_contexts=args.assert_n_contexts
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


num_plots = len(experiment_dirs)
num_cols = min(len(experiment_dirs), 3)  # Adjust this as needed for a better layout
num_rows = num_plots // num_cols + (num_plots % num_cols > 0)


fig, axs = plt.subplots(num_rows, num_cols, figsize=(interval_figsize_x * num_cols, interval_figsize_y * num_rows))

if num_cols == 1:
    axs = [axs]
else:
    axs = axs.flatten()

all_ipsative_corrs_str = get_all_ipsative_corrs_str()

from collections import defaultdict
family_data = defaultdict(list)

for plt_i, experiment_dir in enumerate(experiment_dirs):

    if bar_plots:

        plt.subplots_adjust(left=left_adjust, top=0.90, bottom=0.5, hspace=0.8)
        xs = [x_label_map.get(x, x) for x in models]

        scores = np.array([[data[experiment_dir][model][seed_str][metric] for seed_str in seed_strings] for model in models])

        assert len(models) == len(scores)

        for model, m_scores in zip(models, scores):
            family_data[model_2_family(model)].append(m_scores)

        ys = scores.mean(axis=1)

        n_msgs = models

        # default shape (10 pvq values, 50 sim participants), this is used in case a model is not found
        default_nans = np.empty((10, 50))
        default_nans.fill(np.nan)
        all_corrs = np.array([[
            data[experiment_dir][n_msg][seed_str].get(
                all_ipsative_corrs_str, default_nans
            ) for seed_str in seed_strings
        ] for n_msg in n_msgs])

        all_corrs = all_corrs.mean(2)  # mean over context pairs

        # SI over what -> seed and personas
        all_corrs = all_corrs.reshape(len(n_msgs), -1)

        tick_len_se = np.array([st.sem(a) for a in all_corrs])

        scores = all_corrs

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
                with open("PLOSONE/tolkien_ipsative_curve_cache.json", "w") as f:
                    json.dump(tolkien_ipsative_curve, f)

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

        axs[plt_i].set_ylim(min_y, max_y)
        axs[plt_i].set_xticklabels([x_label_map.get(m, m) for m in models], rotation=rotatation_x_labels, fontsize=xticks_fontsize, ha="right", rotation_mode="anchor")
        axs[plt_i].set_yticklabels(map(lambda x: np.round(x, round_y_lab), axs[plt_i].get_yticks()), fontsize=yticks_fontsize)

        axs[plt_i].set_ylabel(y_label, fontsize=y_label_fontsize)
        if x_label is not None:
            axs[plt_i].set_xlabel(x_label, fontsize=x_label_fontsize)

        if title:
            axs[plt_i].set_title(title, fontsize=title_fontsize)

        if x_label:
            axs[plt_i].set_xlabel(x_label, fontsize=x_label_fontsize)

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

            # default shape (10 pvq values, 50 sim participants), this is used in case a model is not found
            default_nans = np.empty((10, 50))
            default_nans.fill(np.nan)

            all_corrs = np.array([data[experiment_dir][model][msg].get(all_ipsative_corrs_str, default_nans) for msg in seed_strings])
            all_corrs = np.mean(all_corrs, axis=1)  # average over context pairs

            print(f"{model}: {ys}")

            family = model_2_family(model)
            # linestyle = family_2_linestyle[family]
            linestyle = "-"

            xs = [x.replace("/_seed", "") for x in xs]
            xs = [x_label_map.get(x, x) for x in xs]

            # ugly patch to make colors nicer
            if "gpt" in model:
                c = "black"
            elif "phi-1" in model:
                c = "lightgreen"
            elif "phi-2" in model:
                c = "teal"
            else:
                c = None

            line, = axs[plt_i].plot(xs, ys, label=x_label_map.get(model, model), linestyle=linestyle, color=c)
            line_color = line.get_color()
            axs[plt_i].fill_between(xs, ys - shade_len, ys + shade_len, alpha=0.3, color=line_color)

        max_y = 0.8 if metric == "Rank-Order" else 1.0
        axs[plt_i].set_ylim(-0.1, max_y)
        axs[plt_i].set_xlim(0, len(xs) - 1)
        axs[plt_i].set_ylabel(y_label, fontsize=y_label_fontsize)

        if x_label is not None:
            axs[plt_i].set_xlabel(x_label, fontsize=x_label_fontsize)

        # axs[plt_i].set_title(experiment_dir.replace("sim_conv_", "").replace("_seeds", ""))
        if add_legend:
            axs[plt_i].legend(bbox_to_anchor=(1.04, 1), loc="best", fontsize=legend_fontsize)

        axs[plt_i].set_xticklabels(xs, rotation=rotatation_x_labels, fontsize=xticks_fontsize, rotation_mode="anchor")
        axs[plt_i].set_yticklabels(map(lambda x: np.round(x, round_y_lab), axs[plt_i].get_yticks()), fontsize=yticks_fontsize)

        plt.subplots_adjust(left=0.1, top=0.95, bottom=0.2, hspace=0.8)

# Hide any unused subplots
for j in range(plt_i + 1, num_rows * num_cols):
    axs[j].axis('off')

plt.tight_layout()

fig_path = f'PLOSONE/data_analysis/visualizations/{figure_name}.pdf'
print(f"save to: {fig_path}")
plt.savefig(fig_path)

if not args.no_show:
    plt.show()  # Sh
plt.close()


