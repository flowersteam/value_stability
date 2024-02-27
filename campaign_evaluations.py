import glob
import subprocess
import json
import os
from itertools import chain
import numpy as np
import matplotlib.pyplot as plt
import hashlib
import checksumdir
import inspect
import scipy.stats as st
from termcolor import cprint

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
    elif "dummy":
        return "dummy"
    else:
        raise ValueError(f"Unkwown model family for model {model}.")


family_2_color = {
    "LLaMa-2": "blue",
    "Mixtral": "orange",
    "Mistral": "green",
    "Phi": "red",
    "Qwen": "purple",
    "dummy": "black"
}

family_2_linestyle = {
    "LLaMa-2": ":",
    "Mixtral": "-",
    "Mistral": "dashdot",
    "Phi":  (0, (3, 5, 1, 5, 1, 5)),
    "Qwen": "--",
    "dummy": "-"
}


def legend_without_duplicate_labels(ax, loc="best", title=None, legend_loc=None):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    # axs[plt_i].legend(bbox_to_anchor=legend_loc, loc="best")
    ax.legend(*zip(*unique), loc=loc, title=title, fontsize=legend_fontsize, title_fontsize=legend_fontsize, bbox_to_anchor=legend_loc)

def get_all_ipsative_corrs_str(default_profile):

    if default_profile is not None:
        all_ipsative_corrs_str = "All_Ipsative_corrs"
    else:
        all_ipsative_corrs_str = "All_Ipsative_corrs_default_profile"

    return all_ipsative_corrs_str


def run_analysis(eval_script_path, data_dir, assert_n_contexts=None, insert_dummy_participants=False, default_profile=None, paired_data_dir=None, RO_neutral=False, RO_neutral_data_dir=None, no_ips=False):
    # run evaluation script
    command = f"python {eval_script_path} --result-json-stdout {'--assert-n-dirs ' + str(assert_n_contexts) if assert_n_contexts else ''} {'--insert-dummy' if insert_dummy_participants else ''} {f'--default-profile {default_profile}' if default_profile is not None else ''} {data_dir}/*/* {f'--paired-dirs {paired_data_dir}/*/*' if paired_data_dir is not None else ''} {f'--neutral-ranks --neutral-dir {RO_neutral_data_dir}' if RO_neutral else ''} {'--no-ips' if no_ips else ''}"
    # print("Command: ", command)
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    if stderr: print("Error:", stderr.decode())

    # parse json outputs
    results = json.loads(stdout)

    all_ipsative_corrs_str = get_all_ipsative_corrs_str(default_profile)
    results[all_ipsative_corrs_str] = np.array(results[all_ipsative_corrs_str])

    return results

all_data_dirs = []

# Define the models
models = [
    "llama_2_7b",
    "llama_2_13b",
    "llama_2_70b",  # 2 gpu
    "llama_2_7b_chat",
    "llama_2_13b_chat",
    # "llama_2_70b_chat",  # 2 gpu
    "Mistral-7B-v0.1",
    "Mistral-7B-Instruct-v0.1",
    "Mistral-7B-Instruct-v0.2",
    "zephyr-7b-beta",
    "Mixtral-8x7B-v0.1-4b",  # 6h
    "Mixtral-8x7B-Instruct-v0.1-4b",  # 6h
    "Mixtral-8x7B-v0.1",
    "Mixtral-8x7B-Instruct-v0.1",
    "phi-1",
    "phi-2",
    "Qwen-7B",
    "Qwen-14B",
    "Qwen-72B",
    # "dummy"
]

plot_models = [
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
]

models_mmlu = {
    "llama_2_7b": 0.4687,
    "llama_2_13b": 0.5577,
    "llama_2_70b": 0.6993,  # 2 gpu
    "llama_2_7b_chat": 0.4832,
    "llama_2_13b_chat": 0.5464,
    "llama_2_70b_chat": 0.6391,  # 2 gpu
    "Mistral-7B-v0.1": 0.6416,
    "Mistral-7B-Instruct-v0.1": 0.5538,
    "Mistral-7B-Instruct-v0.2": 0.6078,
    "zephyr-7b-beta": 0.6106,
    # "Mixtral-8x7B-v0.1-4b": np.nan,
    # "Mixtral-8x7B-Instruct-v0.1-4b": np.nan,
    "Mixtral-8x7B-v0.1": 0.7188,
    "Mixtral-8x7B-Instruct-v0.1": 0.714,
    "phi-2": 0.5811,
    "phi-1.5": 0.4389,
    "phi-1": None,
}

x_label_map = {
    "dummy": "random",
    "Mixtral-8x7B-v0.1-4b": "Mixtral-Base-4b",
    "Mixtral-8x7B-Instruct-v0.1-4b": "Mixtral-Instruct-4b",
    "Mixtral-8x7B-v0.1": "Mixtral-Base",
    "Mixtral-8x7B-Instruct-v0.1": "Mixtral-Instruct",
    "Mistral-7B-v0.1": "Mistral-Base",
    "Mistral-7B-Instruct-v0.1": "Mistral-Instruct-v0.1",
    "Mistral-7B-Instruct-v0.2": "Mistral-Instruct-v0.2",
    "llama_2_7b":  "LLaMa_2_7b",
    "llama_2_13b": "LLaMa_2_13b",
    "llama_2_70b": "LLaMa_2_70b",
    "llama_2_7b_chat": "LLaMa_2_7b_chat",
    "llama_2_13b_chat": "LLaMa_2_13b_chat",
    "llama_2_70b_chat": "LLaMa_2_70b_chat",
    "phi-2": "Phi-2",
    "phi-1": "Phi-1",

}
x_label_map = {**x_label_map, **{k:k.replace("_msgs", "") for k in ["1_msgs", "3_msgs", "5_msgs", "7_msgs", "9_msgs"]}}
# Define the results directory
# sim conv

add_legend = False
add_title = False

bars_as_plot = False

label_ = None

experiment_dirs = [
    "sim_conv_pvq_permutations_msgs",
    # "sim_conv_pvq_tolkien_characters_seeds",
    # "sim_conv_pvq_famous_people_seeds",
    # "sim_conv_pvq_tolkien_characters_seeds_NO_SYSTEM",
    # "sim_conv_tolkien_donation_tolkien_characters_seeds",
]
if "permutations_msgs" in experiment_dirs[0]:
    seed_strings = [f"{i}_msgs/_seed" for i in range(1, 10, 2)]  # msgs (show trends
    # seed_strings = ["3_msgs/_seed"] # ips (only n=3)
else:
    seed_strings = [f"{i}_seed" for i in range(1, 10, 2)]
    # seed_strings = [f"{i}_seed" for i in range(3, 10, 2)]

add_tolkien_ipsative_curve = True
bar_plots = False

# metric = "Mean-Level"
# metric = "Rank-Order"
metric = "Ipsative"


# figure_name = "tolk_ro_t"
# figure_name = "paired_tolk_ro_uni"
# figure_name = "paired_tolk_ro_ben"
# figure_name = "paired_tolk_ro_pow"
# figure_name = "paired_tolk_ro_ach"
# figure_name = "fam_ro_t"
# figure_name = "don_t"
# figure_name = "bag_t"
# figure_name = "no_pop_ips"
# figure_name = "tolk_ro_msgs"
# figure_name = "tolk_ro_msgs_neutral"
# figure_name = "tolk_ips_msgs"
# figure_name = "no_pop_msgs"
figure_name = "tolk_ips_msgs_default_prof"
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

default_profile = None

add_tolkien_ro_curve = False
add_tolkien_ipsative_curve = False

left_adjust = None
paired_dir = None
y_label = None

RO_neutral = False

if figure_name == "no_pop_msgs":
    experiment_dirs = ["sim_conv_pvq_permutations_msgs"]
    seed_strings = [f"{i}_msgs/_seed" for i in range(1, 10, 2)]  # msgs (show trends

    add_tolkien_ipsative_curve = True
    bar_plots = False
    models = plot_models
    metric = "Ipsative"
    human_change_xloc = -1.0
    msgs_ro_tolk = False

    min_y, max_y = -0.1, 1.0  # IPS
    legend_fontsize = 22
    xticks_fontsize = 20
    y_label_fontsize = 40
    x_label_fontsize = 30
    xticks_fontsize = 20
    yticks_fontsize = 20
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

elif figure_name == "tolk_ro_t":

    experiment_dirs = ["sim_conv_pvq_tolkien_characters_seeds"]

    seed_strings = [f"{i}_seed" for i in range(1, 10, 2)]

    add_tolkien_ipsative_curve = False
    bar_plots = True
    add_legend = True
    add_title = True
    metric = "Rank-Order"
    human_change_xloc = 6.8
    msgs_ro_tolk = False
    show_human_change = True
    legend_fontsize = 22
    rotatation_x_labels = 90

    xticks_fontsize = 15
    yticks_fontsize = 18

    min_y, max_y = -0.1, 0.8  # RO

elif figure_name.startswith("paired_tolk_ro"):

    if figure_name.endswith("uni"):
        value_to_pair = "Universalism"
    elif figure_name.endswith("ben"):
        value_to_pair = "Benevolence"
    elif figure_name.endswith("pow"):
        value_to_pair = "Power"
    elif figure_name.endswith("ach"):
        value_to_pair = "Achievement"
    else:
        raise ValueError(f"Undefined figure name: {figure_name}")

    # y_label = f"Rank-Order stability\n{value_to_pair}-Donation"
    y_label = f"Rank-Order stability\nwith donation"

    experiment_dirs = ["sim_conv_pvq_tolkien_characters_seeds"]
    paired_dir = "sim_conv_tolkien_donation_tolkien_characters_seeds"

    seed_strings = [f"{i}_seed" for i in range(1, 10, 2)]

    add_tolkien_ipsative_curve = False
    bar_plots = True

    if value_to_pair == "Universalism":
        add_legend = True
        legend_fontsize = 20
    else:
        add_legend = False

    add_title = False
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


elif figure_name == "fam_ro_t":

    experiment_dirs = ["sim_conv_pvq_famous_people_seeds"]

    seed_strings = [f"{i}_seed" for i in range(1, 10, 2)]

    add_tolkien_ipsative_curve = False
    bar_plots = True
    add_legend = False
    add_title = True
    metric = "Rank-Order"
    human_change_xloc = 6.8
    msgs_ro_tolk = False

    show_human_change = True
    rotatation_x_labels = 90

    xticks_fontsize = 15
    yticks_fontsize = 18

    min_y, max_y = -0.1, 0.8  # RO

elif figure_name == "don_t":
    experiment_dirs = ["sim_conv_tolkien_donation_tolkien_characters_seeds"]

    seed_strings = [f"{i}_seed" for i in range(1, 10, 2)]

    add_tolkien_ipsative_curve = False
    bar_plots = True
    add_legend = False
    add_title = True
    metric = "Rank-Order"
    human_change_xloc = 6.8
    msgs_ro_tolk = False
    rotatation_x_labels = 90

    xticks_fontsize = 15
    yticks_fontsize = 18

    min_y, max_y = -0.1, 0.8  # RO

elif figure_name == "bag_t":
    experiment_dirs = ["sim_conv_tolkien_bag_tolkien_characters_seeds"]

    seed_strings = [f"{i}_seed" for i in range(1, 10, 2)]

    add_tolkien_ipsative_curve = False
    bar_plots = True
    add_legend = False
    add_title = True
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
    add_title = True
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
    raise ValueError(f"Unknown figure name {figure_name}.")

if y_label is None:
    y_label = metric + " stability (r)"

if add_tolkien_ipsative_curve:
    with open("tolkien_ipsative_curve_cache.json", "r") as f:
        tolkien_ipsative_curve = json.load(f)

if add_tolkien_ro_curve:
    with open("tolkien_ro_curve_cache.json", "r") as f:
        tolkien_ro_curve = json.load(f)

confidence = 0.95

stab_mmlu_scatter = False  # overrides other plots

insert_dummy_participants = False

if insert_dummy_participants:
    print("Inserting dummy participants.")

assert_n_contexts = None
# assert_n_contexts = 6
assert_n_contexts = 5
# assert_n_contexts = 4

if assert_n_contexts:
    cprint(f"Asserting {assert_n_contexts} contexts.", "green")

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

            if experiment_dir == "mmlu":
                data[experiment_dir][model][seed_str] = {"Accuracy": models_mmlu[model]}
                continue

            data_dir = os.path.join("results", experiment_dir, model, seed_str)
            if paired_dir:
                paired_data_dir = os.path.join("results", paired_dir, model, seed_str)
            else:
                paired_data_dir = None

            if RO_neutral:
                RO_neutral_data_dir = os.path.join("results", RO_neutral_dir, seed_str)
            else:
                RO_neutral_data_dir = None

            if len(glob.glob(data_dir+"/*/*/*.json")) < 2:
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
                     str(assert_n_contexts), str(insert_dummy_participants),
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
                        eval_script_path=eval_script_path, data_dir=data_dir, assert_n_contexts=assert_n_contexts,
                        insert_dummy_participants=insert_dummy_participants, default_profile=default_profile,
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

        # axs[plt_i].scatter(models, [models_mmlu[m] for m in models], marker="x", s=5)

    if stab_mmlu_scatter:
        models_to_scatter = list(models_mmlu.keys())
        stabs = np.array(
            [[data[experiment_dir][model][s_][metric] for s_ in seed_strings] for model in models_to_scatter]
        ).mean(axis=1)
        mmlus = [models_mmlu[m] for m in models_to_scatter]

        axs[plt_i].set_ylabel(y_label, fontsize=y_label_fontsize)
        axs[plt_i].set_xlabel("MMLU score", fontsize=20)
        for m,s, mod in zip(mmlus, stabs, models_to_scatter):
            axs[plt_i].text(m+0.005, s+0.005, mod, fontsize=8)

        def m2c(m):
            if "llama" in m:
                return "b"
            elif "Mistral" in m or "zephyr" in m:
                return "g"
            elif "Mixtral" in m:
                return "r"
            else:
                raise ValueError(f"Undefined color for model {m}.")

        colors = [m2c(m) for m in models_to_scatter]

        axs[plt_i].scatter(x=mmlus, y=stabs, c=colors)
        # axs[plt_i].text(-0.8, human_change_10_12[metric] + 0.01, "Human value stability between ages 10 and 12", fontsize=6, color=human_data_color)

        # # Plotting the regression line
        # from scipy.stats import linregress
        # slope, intercept, r_value, p_value, std_err = linregress(mmlus, stabs)
        # print(f"r:{r_value}", f"p:{p_value}, err:{std_err}")
        # def regression_line(x): return slope * x + intercept
        #
        # line_x = np.linspace(min(mmlus), max(mmlus), 100)
        # line_y = regression_line(line_x)
        # axs[plt_i].plot( line_x, line_y, color='black', linewidth=2)

    elif bar_plots:

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

        ys = scores.mean(axis=1)

        # get the right side of the CI
        if "sim_conv_pvq_permutations_msgs" in experiment_dir:
            assert len(seed_strings) == 1  # you should use plots, not bars
            # [n_modelx, cont_pairs, pop_size]
            all_corrs = np.array([data[experiment_dir][model][seed_strings[0]][all_ipsative_corrs_str] for model in models])
            all_corrs = all_corrs.mean(1)  # mean over pairs

            c2 = np.array([st.t.interval(confidence, len(a) - 1, loc=np.mean(a), scale=st.sem(a))[1] for a in all_corrs])
            tick_len = c2 - ys  # half the conf interval

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
            tick_len = c2 - ys  # half the conf interval

        else:
            c2 = np.array([st.t.interval(confidence, len(a) - 1, loc=np.mean(a), scale=st.sem(a))[1] for a in scores])
            tick_len = c2 - ys  # half the conf interval

        if np.isnan(ys).all():
            raise ValueError("All models are nan.")

        if bars_as_plot:
            # used for msgs
            axs[plt_i].plot(xs, ys, label=label_)
            axs[plt_i].fill_between(xs, ys - tick_len, ys + tick_len, alpha=0.3)

            # if default_profile is not None and figure_name == "tolk_ips_msgs_default_prof":
                # # load ro
                # if add_tolkien_ro_curve:
                #     cprint("Loading Rank-order Tolkien Mixtral-Instruct stability from CACHE", "red")
                #
                #     xs = np.array(tolkien_ro_curve["xs"])
                #     xs = [x_label_map.get(x, x) for x in xs]
                #     ys = np.array(tolkien_ro_curve["ys"])
                #     shade_len = np.array(tolkien_ro_curve["tick_len"])
                #
                #     lab_ = "Rank-Order stability\n  (between contexts)"
                #     col_ = "black"
                #     axs[plt_i].plot(xs, ys, label=lab_, color=col_)
                #     axs[plt_i].fill_between(xs, ys - shade_len, ys + shade_len, alpha=0.3, color=col_)
                #
                # if add_tolkien_ipsative_curve:
                #     cprint("Loading Ipsative Tolkien Mixtral-Instruct stability from CACHE", "red")
                #
                #     xs = np.array(tolkien_ipsative_curve["xs"])
                #     xs = [x_label_map.get(x, x) for x in xs]
                #     ys = np.array(tolkien_ipsative_curve["ys"])
                #     shade_len = np.array(tolkien_ipsative_curve["tick_len"])
                #
                #     lab_ = "Ipsative stability (between contexts)"
                #     col_ = "brown"
                #     axs[plt_i].plot(xs, ys, label=lab_, color=col_, zorder=0)
                #     axs[plt_i].fill_between(xs, ys - shade_len, ys + shade_len, alpha=0.3, color=col_, zorder=0)

            if metric == "Ipsative" and figure_name == "tolk_ips_msgs":
                tolkien_ipsative_curve = {
                    "xs": list(xs),
                    "ys": list(ys),
                    "tick_len": list(tick_len),
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
                axs[plt_i].bar(xs, ys, yerr=tick_len)

                if metric == "Rank-Order" and figure_name == "tolk_ro_msgs":
                    tolkien_ro_curve = {
                        "xs": list(xs),
                        "ys": list(ys),
                        "tick_len": list(tick_len),
                    }

                    cprint("SAVING Rank-order Tolkien Mixtral-Instruct stability to CACHE", "red")
                    with open("tolkien_ro_curve_cache.json", "w") as f:
                        json.dump(tolkien_ro_curve, f)

            else:
                cs = [family_2_color[model_2_family(x)] for x in xs]
                labs = [model_2_family(x) for x in xs]

                axs[plt_i].bar(xs, ys, yerr=tick_len, color=cs, label=labs)

        axs[plt_i].set_ylim(min_y, max_y)
        axs[plt_i].set_xticklabels([x_label_map.get(m, m) for m in models], rotation=rotatation_x_labels, fontsize=xticks_fontsize)
        axs[plt_i].set_yticklabels(map(lambda x: np.round(x, round_y_lab), axs[plt_i].get_yticks()), fontsize=yticks_fontsize)

        # axs[plt_i].set_title(experiment_dir.replace("sim_conv_", "").replace("_seeds", ""))

        axs[plt_i].set_ylabel(y_label, fontsize=y_label_fontsize)

        if add_title:
            if "donation" in experiment_dir:
                axs[plt_i].set_title("Donation stability of fictional characters", fontsize=title_fontsize)
            else:
                if "tolkien" in experiment_dir:
                    axs[plt_i].set_title("Personal value stability of fictional characters with PVQ", fontsize=title_fontsize)
                elif "famous" in experiment_dir:
                    axs[plt_i].set_title("Personal value stability of real world personas with PVQ", fontsize=title_fontsize)


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
            linestyle = family_2_linestyle[family]
            linestyle = "-"

            xs = [x.replace("/_seed", "") for x in xs]
            xs = [x_label_map.get(x,x) for x in xs]
            axs[plt_i].plot(xs, ys, label=x_label_map.get(model, model), linestyle=linestyle)
            axs[plt_i].fill_between(xs, ys - shade_len, ys + shade_len, alpha=0.3)

        max_y = 0.8 if metric == "Rank-Order" else 1.0
        axs[plt_i].set_ylim(-0.1, max_y)
        axs[plt_i].set_xlim(0, len(seed_strings) - 1)
        axs[plt_i].set_ylabel(y_label, fontsize=y_label_fontsize)
        axs[plt_i].set_xlabel("Simulated conversation length (n)", fontsize=x_label_fontsize)
        # axs[plt_i].set_title(experiment_dir.replace("sim_conv_", "").replace("_seeds", ""))
        axs[plt_i].legend(bbox_to_anchor=(1.04, 1), loc="best", fontsize=legend_fontsize)

        axs[plt_i].set_xticklabels(axs[plt_i].get_xticks(), rotation=rotatation_x_labels, fontsize=xticks_fontsize)
        axs[plt_i].set_yticklabels(map(lambda  x: np.round(x, round_y_lab), axs[plt_i].get_yticks()), fontsize=yticks_fontsize)

        plt.subplots_adjust(left=0.1, top=0.95, bottom=0.2, hspace=0.8)

# Hide any unused subplots
for j in range(plt_i + 1, num_rows * num_cols):
    axs[j].axis('off')

plt.tight_layout()

# fig_path = f'visualizations/{figure_name}.png'
fig_path = f'visualizations/{figure_name}.svg'
print(f"save to: {fig_path}")
plt.savefig(fig_path)
plt.show()  # Sh
plt.close()
