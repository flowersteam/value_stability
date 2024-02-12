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
    if "llama_2" in model:
        return "LLaMa-2"
    elif "Mixtral" in model:
        return "Mixtral"
    elif "Mistral" in model or "zephyr" in model:
       return "Mistral"
    elif "phi" in model:
        return "Phi"
    elif "Qwen" in model:
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


def run_analysis(eval_script_path, data_dir, assert_n_contexts=None, insert_dummy_participants=False):
    # run evaluation script
    command = f"python {eval_script_path} --result-json-stdout {'--assert-n-dirs ' + str(assert_n_contexts) if assert_n_contexts else ''} {'--insert-dummy' if insert_dummy_participants else ''} {data_dir}/*/*"
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    if stderr: print("Error:", stderr.decode())

    # parse json outputs
    results = json.loads(stdout)
    results["All_Ipsative_corrs"]=np.array(results["All_Ipsative_corrs"])

    return results

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
# figure_name = "fam_ro_t"
figure_name = "don_t"
# figure_name = "no_pop_ips"
# figure_name = "tolk_ro_msgs"
# figure_name = "tolk_ips_msgs"
# figure_name = "no_pop_msgs"

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

show_human_change = False
legend_loc = None

if figure_name == "no_pop_msgs":
    experiment_dirs = ["sim_conv_pvq_permutations_msgs"]
    seed_strings = [f"{i}_msgs/_seed" for i in range(1, 10, 2)]  # msgs (show trends

    add_tolkien_ipsative_curve = True
    bar_plots = False
    models = plot_models
    metric = "Ipsative"
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

elif figure_name == "no_pop_ips":
    experiment_dirs = ["sim_conv_pvq_permutations_msgs"]
    seed_strings = ["3_msgs/_seed"]  # ips (only n=3)

    add_tolkien_ipsative_curve = False
    bar_plots = True
    add_legend = True
    metric = "Ipsative"
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
    msgs_ro_tolk = False
    show_human_change = True
    legend_fontsize = 22
    rotatation_x_labels = 90

    xticks_fontsize = 15
    yticks_fontsize = 18

    min_y, max_y = -0.1, 0.8  # RO


elif figure_name == "fam_ro_t":

    experiment_dirs = ["sim_conv_pvq_famous_people_seeds"]

    seed_strings = [f"{i}_seed" for i in range(1, 10, 2)]

    add_tolkien_ipsative_curve = False
    bar_plots = True
    add_legend = False
    add_title = True
    metric = "Rank-Order"
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
    interval_figsize_x = 14
    interval_figsize_y = 7

    xticks_fontsize = 25
    yticks_fontsize = 25
    y_label_fontsize = 35
    x_label_fontsize = 30

    min_y, max_y = 0.25, 0.5  # RO

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

    min_y, max_y = -0.1, 1  # IPS
else:
    raise ValueError(f"Unknown figure name {figure_name}.")



if add_tolkien_ipsative_curve:
    with open("tolkien_ipsative_curve_cache.json", "r") as f:
        tolkien_ipsative_curve = json.load(f)

confidence = 0.95

stab_mmlu_scatter = False  # overrides other plots

insert_dummy_participants = False

if insert_dummy_participants:
    print("Inserting dummy participants.")

assert_n_contexts = None
# assert_n_contexts = 6
# assert_n_contexts = 5
# assert_n_contexts = 4

if assert_n_contexts:
    cprint(f"Asserting {assert_n_contexts} contexts.", "green")

# prefix = "results_pvq_sim_conv_famous_people"
# prefix = "results_ult_sim_conv_famous_people"


# # # sim conv - ultimatum
# data_dirs = [
#     "results_ultimatum_sim_conv_v2_perm",
# ]
# assert_n_contexts = 5
# prefix = "results_pvq_sim_conv_tolkien_characters"


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

            if len(glob.glob(data_dir+"/*/*/*.json")) < 2:
                print(f"No evaluation found at {data_dir}.")
                # no evaluations
                eval_data = dict(zip(["Mean-Level", "Rank-Order", "Ipsative"], [np.nan, np.nan, np.nan]))

            else:
                # compute hash
                eval_script_path = "./visualization_scripts/data_analysis.py"
                with open(eval_script_path, 'rb') as file_obj: eval_script = str(file_obj.read())
                hash = hashlib.sha256("-".join(
                    [eval_script, inspect.getsource(run_analysis), checksumdir.dirhash(data_dir), str(assert_n_contexts), str(insert_dummy_participants)]
                ).encode()).hexdigest()
                cache_path = f".cache/{hash}.json"

                # check for cache
                if os.path.isfile(cache_path):
                    with open(cache_path) as f:
                        print("\t\tLoading from cache")
                        eval_data = json.load(f)

                else:
                    print("\t\tEvaluating")
                    eval_data = run_analysis(eval_script_path=eval_script_path, data_dir=data_dir, assert_n_contexts=assert_n_contexts, insert_dummy_participants=insert_dummy_participants)

                with open(cache_path, 'w') as fp:

                    class NumpyEncoder(json.JSONEncoder):
                        def default(self, obj):
                            if isinstance(obj, np.ndarray):
                                return obj.tolist()
                            return json.JSONEncoder.default(self, obj)

                    json.dump(eval_data, fp, cls=NumpyEncoder)

            data[experiment_dir][model][seed_str] = eval_data.copy()

            metrs_str = {k: np.round(v, 2) for k,v in data[experiment_dir][model][seed_str].items() if k != "All_Ipsative_corrs"}
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

for plt_i, experiment_dir in enumerate(experiment_dirs):

    if show_human_change:

        if metric == "Rank-Order":
            xloc = 6.8
        else:
            xloc = -1.0

        axs[plt_i].axhline(y=human_change_10_12[metric], color=human_data_color, linestyle=':', zorder=0)
        axs[plt_i].text(xloc, human_change_10_12[metric] + 0.01, "Human value stability between ages 10 and 12",
                        fontsize=human_data_fontsize, color=human_data_color)

        axs[plt_i].axhline(y=human_change_20_28[metric], color=human_data_color, linestyle=':', zorder=0)
        axs[plt_i].text(xloc, human_change_20_28[metric] + 0.01, "Human value stability between ages 20 and 28",
                        fontsize=human_data_fontsize, color=human_data_color)

        # axs[plt_i].scatter(models, [models_mmlu[m] for m in models], marker="x", s=5)

    if stab_mmlu_scatter:
        models_to_scatter = list(models_mmlu.keys())
        stabs = np.array(
            [[data[experiment_dir][model][s_][metric] for s_ in seed_strings] for model in models_to_scatter]
        ).mean(axis=1)
        mmlus = [models_mmlu[m] for m in models_to_scatter]

        axs[plt_i].set_ylabel(f"{metric}", fontsize=y_label_fontsize)
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

        plt.subplots_adjust(top=0.90, bottom=0.5, hspace=0.8)
        xs = models
        scores = np.array([[data[experiment_dir][model][seed_str][metric] for seed_str in seed_strings] for model in models])
        ys = scores.mean(axis=1)

        # get the right side of the CI
        if "sim_conv_pvq_permutations_msgs" in experiment_dir:
            assert len(seed_strings) == 1 # you should use plots, not bars
            # [n_modelx, cont_pairs, pop_size]
            all_corrs = np.array([data[experiment_dir][model][seed_strings[0]]["All_Ipsative_corrs"] for model in models])
            all_corrs = all_corrs.mean(1) # mean over pairs

            c2 = np.array([st.t.interval(confidence, len(a) - 1, loc=np.mean(a), scale=st.sem(a))[1] for a in all_corrs])
            tick_len = c2 - ys  # half the conf interval

        elif metric == "Ipsative":
            n_msgs = models
            # [n_msgs, n_seeds, n_pairs, n_personas]
            all_corrs = np.array([[
                data[experiment_dir][n_msg][seed_str]["All_Ipsative_corrs"] for seed_str in seed_strings
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
            axs[plt_i].plot(xs, ys)
            axs[plt_i].fill_between(xs, ys - tick_len, ys + tick_len, alpha=0.3)

            if metric == "Ipsative":
                tolkien_ipsative_curve = {
                    "xs": list(xs),
                    "ys": list(ys),
                    "tick_len": list(tick_len),
                }

                cprint("SAVING Ipsative Rank-order Tolkien Mixtral-Instruct stability to CACHE", "red")
                with open("tolkien_ipsative_curve_cache.json", "w") as f:
                    json.dump(tolkien_ipsative_curve, f)

        else:
            if msgs_ro_tolk:
                axs[plt_i].bar(xs, ys, yerr=tick_len)

            else:
                cs = [family_2_color[model_2_family(x)] for x in xs]
                labs = [model_2_family(x) for x in xs]

                axs[plt_i].bar(xs, ys, yerr=tick_len, color=cs, label=labs)


        axs[plt_i].set_ylim(min_y, max_y)
        axs[plt_i].set_xticklabels([x_label_map.get(m, m) for m in models], rotation=rotatation_x_labels, fontsize=xticks_fontsize)
        axs[plt_i].set_yticklabels(map(lambda x: np.round(x, 1), axs[plt_i].get_yticks()), fontsize=yticks_fontsize)

        # axs[plt_i].set_title(experiment_dir.replace("sim_conv_", "").replace("_seeds", ""))

        axs[plt_i].set_ylabel(metric + " stability (r)", fontsize=y_label_fontsize)

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
            legend_without_duplicate_labels(axs[plt_i], loc="best", title="LLM families", legend_loc=legend_loc)

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
            all_corrs = np.array([data[experiment_dir][model][msg]["All_Ipsative_corrs"] for msg in seed_strings])
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
        axs[plt_i].set_ylabel(metric + " stability (r)", fontsize=y_label_fontsize)
        axs[plt_i].set_xlabel("Simulated conversation length (n)", fontsize=x_label_fontsize)
        # axs[plt_i].set_title(experiment_dir.replace("sim_conv_", "").replace("_seeds", ""))
        axs[plt_i].legend(bbox_to_anchor=(1.04, 1), loc="best", fontsize=legend_fontsize)

        axs[plt_i].set_xticklabels(axs[plt_i].get_xticks(), rotation=rotatation_x_labels, fontsize=xticks_fontsize)
        axs[plt_i].set_yticklabels(map(lambda  x: np.round(x,1 ), axs[plt_i].get_yticks()), fontsize=yticks_fontsize)

        plt.subplots_adjust(left=0.1, top=0.95, bottom=0.2, hspace=0.8)

# Hide any unused subplots
for j in range(plt_i + 1, num_rows * num_cols):
    axs[j].axis('off')

plt.tight_layout()

# plt.savefig(f'visualizations/plot.png')
fig_path =  f'visualizations/{figure_name}.svg'
print(f"save to: {fig_path}")
plt.savefig(fig_path)
plt.show()  # Sh
plt.close()
