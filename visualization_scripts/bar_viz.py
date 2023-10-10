import os
import json
import warnings

import matplotlib.pyplot as plt
import re
import numpy as np
from termcolor import colored
import scipy.stats as stats
import itertools
from scipy.stats import tukey_hsd, shapiro, mannwhitneyu, kstest, sem
import scikit_posthocs as sp
import pandas as pd
import pingouin as pg
from itertools import combinations
from scipy.stats import pearsonr, spearmanr

def is_strictly_increasing(lst):
    return all(x <= y for x, y in zip(lst, lst[1:]))


def dir_to_label(directory):

    if "format_chat_simulate_conv" in directory:
        label = extract_value(directory, "_simulate_conv_")
        label = label[:label.find("____")]

    elif "simulate_conv" in directory:
        label = extract_value(directory, "_simulate_conv_")

    elif "no_profile" in directory:
        label = extract_value(directory, "_format_")

    elif "profile" in directory:
        label = extract_profile(directory)
        label = label.replace("Primary values:", "")

    elif "lotr_character" in directory:
        label = extract_value(directory, "_lotr_character_")
    elif "text_type" in directory:
        label = extract_value(directory, "_text_type_")
    elif "music_expert" in directory:
        label = extract_value(directory, "_music_expert_")
    elif "music_AI_experts" in directory:
        label = extract_value(directory, "_music_expert_")
    elif "hobby" in directory:
        label = extract_value(directory, "_hobby_")
    else:
        label = os.path.basename(directory)

    label = label.rstrip("_").lstrip("_")
    return label


def color_for_label(label):
    label_to_color = {
        "Democracy": "blue",
        "Theocracy": "black",
        # "Communism": "red",
        "Totalitarianism": "gold",
        "Anarchy": "brown",

        # "Christianity": "lightyellow",
        # "Pagan": "gray"
    }

    for k, v in label_to_color.items():
        if k in label:
            return v

    return None


# all_values_ = []
def extract_value(directory, key="_lotr_character_"):
    label = os.path.basename(directory)
    if key in label:
        start_index = label.find(key) + len(key)

    elif "_ntrain_" in label:
        start_index = label.find("ntrain_") + len("ntrain_") + 1

    else:
        start_index = 0

    if "__2023" in label:
        end_index = label.find("__2023")
    elif "_2023" in label:
        end_index = label.find("_2023")
    else:
        end_index = len(label)

    label = label[start_index:end_index]

    return label

def extract_profile(directory):
    label = os.path.basename(directory)

    if "_profile_" in label:
        start_index = label.find("_profile_") + len("_profile_")

    elif "_ntrain_" in label:
        start_index = label.find("ntrain_") + len("ntrain_") + 1

    else:
        start_index = 0

    if "_2023" in label:
        end_index = label.find("_2023")
    else:
        end_index = len(label)

    label = label[start_index:end_index]

    return label
def subjects_average(data, subjects_to_average, metric="accuracy"):
    present_subjects = list(data['metrics'].keys())

    # all subjects to average are present
    if all(avg_s in present_subjects for avg_s in subjects_to_average):
        return np.mean([data['metrics'][s][metric] for s in subjects_to_average])
    else:
        return None


def extract_by_key(directory, key="Hobbies"):
    if key is None:
        return os.path.basename(directory)

    pattern = rf'{key}:([^_]+)'
    match = re.search(pattern, directory)
    if match:
        return match.group(1)
    else:
        return 'Unknown'

def print_aggregated_correlation_stats(cs):
    print("Total stats")
    statistics = {
        "Mean": np.mean(cs),
        "Median": np.median(cs),
        "STD": np.std(cs),
        "Min": np.min(cs),
        "Max": np.max(cs),
        "Perc 25": np.percentile(cs, 25),
        "Perc 75": np.percentile(cs, 75),
    }

    headers = "\t".join(statistics.keys())
    values = "\t".join(f"{val:.2f}" for val in statistics.values())

    print(f"{headers}")
    print(f"{values}")

def print_correlation_stats(cs):
    from scipy.stats import kurtosis, skew

    statistics = {
        "Mean": np.mean(cs),
        "Median": np.median(cs),
        "STD": np.std(cs),
        "Min": np.min(cs),
        "Max": np.max(cs),
        "Perc 25": np.percentile(cs, 25),
        "Perc 75": np.percentile(cs, 75),
        "Skew": skew(cs, axis=0, bias=True),
        "Kurtosis": kurtosis(cs, axis=0, bias=True)
    }

    headers = "\t".join(statistics.keys())
    values = "\t".join(f"{val:.2f}" for val in statistics.values())

    print(f"\t{headers}")
    print(f"\t{values}")

    # print("\tAvg Pearson corr:", np.mean(cs))
    # print("\tMedian Pearson corr:", np.median(cs))
    # print()
    # print("\tSTD Pearson corr:", np.std(cs))
    # print("\tMin Pearson corr:", np.min(cs))
    # print("\tMax Pearson corr:", np.max(cs))
    # print()
    # print("\tPerc 25 Pearson corr:", np.percentile(cs, 25))
    # print("\tPerc 75 Pearson corr:", np.percentile(cs, 75))
    # print()
    # print("\tSkew:", skew(cs, axis=0, bias=True))
    # print("\tKurtosis:", kurtosis(cs, axis=0, bias=True))


def cohen_d(data_1, data_2):
    # Compute means
    mean1 = data_1.mean()
    mean2 = data_2.mean()

    # Compute sample variances
    var1 = data_1.var()
    var2 = data_2.var()

    n1 = len(data_1)
    n2 = len(data_2)

    # Compute pooled standard deviation
    s_pooled = np.sqrt(((n1 - 1) * var1 + (n1 - 1) * var2) / (n1 + n2 - 2))

    # Compute Cohen's d
    d = (mean1 - mean2) / s_pooled

    return d


def plot_baseline(data, ax, directory, offset, keys_to_plot=None, subj=None, bar_width=1.0, min_bar_size=0.1, horizontal_bar=False, value_limit=250, all_evaluation_data=None):

    if subj:
        draw_metrics = data['metrics'][subj]

    else:
        # only one subject
        subjects = list(data['metrics'].keys())

        if len(subjects) == 1:
            assert len(list(data['metrics'].values())) == 1
            draw_metrics = list(data['metrics'].values())[0]

        # only one metric
        elif (
                # all the subjects have only one metric
                len(set([len(v.keys()) for v in data['metrics'].values()])) == 1
        ) and (
                # that is the same metric
                len(set([list(v.keys())[0] for v in data['metrics'].values()])) == 1
        ):
            draw_metrics = {}
            for subj, metrics in data['metrics'].items():
                # only one metric
                assert len(metrics.keys()) == 1
                value = list(metrics.values())[0]

                draw_metrics[subj] = value

        else:
            draw_metrics = {}
            for subj, metrics in data['metrics'].items():
                for metric, value in metrics.items():
                    draw_metrics[f"{subj}_{metric}"] = value

        # add averages
        mean_college_perf = subjects_average(subjects_to_average=[
            "college_biology",
            "college_chemistry",
            "college_computer_science",
            "college_mathematics",
            "college_medicine",
            "college_physics",
        ], metric='accuracy', data=data)

        if mean_college_perf is not None:
            draw_metrics["Mean"] = mean_college_perf

        mean_hs_perf = subjects_average(subjects_to_average=[
            "high_school_biology",
            "high_school_chemistry",
            "high_school_computer_science",
            "high_school_european_history",
            "high_school_geography",
            "high_school_mathematics",
            "high_school_physics",
            "high_school_psychology",
            "high_school_us_history",
            "high_school_world_history",
        ], metric='accuracy', data=data)

        if mean_hs_perf is not None:
            draw_metrics["Mean"] = mean_hs_perf

        mean_tom = subjects_average(subjects_to_average=[
            "tomi_second_order_tom",
            "tomi_first_order_tom",
            "tomi_second_order_no_tom",
            "tomi_first_order_no_tom"
        ], metric='accuracy', data=data)

        if mean_tom is not None:
            draw_metrics["Mean ToM"] = mean_tom

        mean_tom_fo = subjects_average(subjects_to_average=[
            "tomi_first_order_tom",
            "tomi_first_order_no_tom"
        ], metric='accuracy', data=data)

        if mean_tom_fo is not None:
            draw_metrics["Mean ToM (fo)"] = mean_tom_fo

    if keys_to_plot is None:
        # use all the keys
        keys = list(draw_metrics.keys())
    else:
        keys = keys_to_plot

    key_indices = {key: i for i, key in enumerate(keys)}

    # values = [draw_metrics[key] for key in keys] # todo: draw_metrics not needed anymore?

    perm_vals = []
    for perm_m in data['per_permutation_metrics']:
        perm_vals.append([perm_m[test_set_name][key] for key in keys])

    perm_vals = np.array(perm_vals)

    if test_set_name == "pvq_male":
        # centering PVQ
        # todo: fix because this mean is not perfect (should be mean of all answers)
        # this is mean of scores - universalism has 5 questions the rest 4
        perm_vals = perm_vals - np.repeat(perm_vals.mean(1)[:, np.newaxis], 10, axis=1)

    values = perm_vals.mean(0)
    # errs = perm_vals.std(0)

    errs = sem(perm_vals)

    label = dir_to_label(directory)

    print("---------------------")
    print(label)
    keys_vals=dict(zip(keys, values))
    sorted_keys = [k for k, v in sorted(keys_vals.items(), key=lambda item: item[1], reverse=True)]
    print("Order:")
    for k_i, k in enumerate(sorted_keys):
        print("\t{}. {} - {:.2f}".format(k_i, k, keys_vals[k]))
    print()

    x_values = [key_indices[key] + offset for key in keys]

    x_values = [v+bar_width/2 for v in x_values]

    # create dummy bars if 0
    v_to_add = []
    x_to_add = []

    for ind, v in enumerate(values):
        if abs(v) < bar_width/2:
            v_to_add.extend([min_bar_size, -min_bar_size])
            x_to_add.extend([x_values[ind], x_values[ind]])

    if v_to_add:
        values = np.array(list(values) + v_to_add)
        x_values = np.array(list(x_values) + x_to_add)
        errs = np.array(list(errs) + [0.0]*len(x_to_add))

    if horizontal_bar:
        # draw a horizontal bar
        ax.barh(x_values, values, xerr=errs, label=label, height=bar_width, color=color_for_label(label))

    else:
        if figure_draw:
            labels = [label] * len(values)

            bar_color_dict = {
                'Conformity': "orange",
                'Tradition': "orange",
                'Benevolence': "green",
                'Universalism': "green",
                'Self-Direction': "blue",
                'Stimulation': "blue",
                'Hedonism': "blue",
                'Achievement': "red",
                'Power': "red",
                'Security': "orange"
            }

            for i, (value, label, key) in enumerate(zip(values, labels, keys)):
                plt.bar(x_values[i], value, label=label, width=bar_width, color=bar_color_dict[key])

        else:
            assert len(errs) == len(values)
            ax.bar(x_values, values, yerr=errs, label=label, width=bar_width, color=color_for_label(label))

    if args.horizontal_bar:
        assert all([-value_limit <= v <= value_limit for v in values])
        ax.set_xlim([-value_limit, value_limit])

        # ax.set_ylabel('Values', fontsize=15)
        ax.set_xlabel('Scores', fontsize=15)

    else:
        # set y-axis limits
        if test_set_name == "pvq_male":
            # ax.set_ylim([-3, 3]) # append
            ax.set_ylim([-2.9, 2.9])

        elif test_set_name == "hofstede":
            # ax.set_ylim([-350, 350]) # append
            ax.set_ylim([-130, 230])

        elif test_set_name == "big5_50":
            ax.set_ylim([0, 55])

        elif test_set_name == "big5_100":
            ax.set_ylim([0, 110])

        else:
            ax.set_ylim([0, max([6, *values])+0.1])

        # ax.set_xlabel('Values', fontsize=30)
        ax.set_ylabel('Scores', fontsize=30)


    if not args.separate_legend:
        if "lotr" in directories[0]:
            ax.legend(bbox_to_anchor=(0.5, 1.2), loc="center", fontsize=20, ncols=5)
        elif "nat_lang_prof" in directories[0]:
            ax.legend(bbox_to_anchor=(0.5, 1.2), loc="center", fontsize=20, ncols=2)
        elif "music" in directories[0]:
            ax.legend(bbox_to_anchor=(0.5, 1.2), loc="center", fontsize=20, ncols=5)
        elif "hobbies":
            ax.legend(bbox_to_anchor=(0.5, 1.2), loc="center", fontsize=20, ncols=3)
        else:
            ax.legend(loc="best", fontsize=20)
        # fig.subplots_adjust(left=0.05, right=0.95, top=0.8, bottom=0.35)
        fig.subplots_adjust(left=0.15, right=0.95, top=0.82, bottom=0.38)

    return keys, sorted_keys, keys_vals

figure_draw = False

if __name__ == '__main__':
    import argparse

    normalized_evaluation_data = []
    notnorm_evaluation_data = []

    parser = argparse.ArgumentParser()
    parser.add_argument('directories', nargs='+', help='directories containing results.json files')
    parser.add_argument('--save', action="store_true")
    parser.add_argument('--separate_legend', action="store_true")
    parser.add_argument('--filename', type=str, default="hobbies_pvq")
    parser.add_argument('--horizontal-bar', '-hb', action="store_true")
    args = parser.parse_args()

    different_distr = [] # value/traits where the anova test said it's different

    keys_to_plot = None
    # keys_to_plot = ['Conformity', 'Tradition', 'Benevolence', 'Universalism', 'Self-Direction', 'Stimulation', 'Hedonism', 'Achievement', 'Power', 'Security']
    # keys_to_plot = ["power_distance", "individualism", "masculinity", "uncertainty_avoidance", "long_term_orientation", "indulgence"]

    bar_width = 0.10
    bar_margin = 1.2

    if figure_draw:
        bar_width = 0.9

    mean_primary_value_alignment = None
    spearman = False
    compare_scores = False

    # fig, ax = plt.subplots(figsize=(15, 6))
    # fig, ax = plt.subplots(figsize=(13, 10))
    fig, ax = plt.subplots(figsize=(15, 10))


    ignore = [
        "religion",
        "tax",
        "vacation",

        # "grammar",
        # "poem",
        # "joke",
        # "history",
        # "chess",
    ]
    args.directories = [d for d in args.directories if not any([i in d for i in ignore])]
    num_dirs = len([d for d in args.directories if os.path.isdir(d)])
    all_bars_width = num_dirs * (bar_width*bar_margin)  # bars with margins

    keys = None

    # chronological order
    directories = args.directories

    for soc_type in ["Hunter-Gatherer", "Horticultural and Pastoral", "Agricultural", "Industrial", "Postindustrial"]:
        for d in directories:
            if soc_type in d:
                directories.append(d)

    if any(["Age" in d for d in directories]):
        # extract the number after Age:
        def sort_by_key(directories, key):
            dir_2_value = {}
            for dir in directories:
                value = extract_by_key(dir, key)
                value = int(value) if value.isdigit() else -1
                dir_2_value[dir] = value

            # sort by value
            sorted_directories = sorted(directories, key=dir_2_value.get)
            return sorted_directories

        directories = sort_by_key(directories, "Age")

    # remove directories which contain substrings from the list
    ignore_patterns = []
    ignore_patterns = ["gen_space", "gen_w_space"]
    print("Ignoring patterns: ", ignore_patterns)

    for substring in ignore_patterns:
        directories = [d for d in directories if substring not in d]

    directories = [d for d in directories if os.path.isfile(os.path.join(d, 'results.json'))]

    if "pvq_test" in directories[0] or "pvq" in directories[0]:
        test_set_name = "pvq_male"
    elif "hofstede" in directories[0]:
        test_set_name = "hofstede"
    elif "big5_50" in directories[0]:
        test_set_name = "big5_50"
    elif "big5_100" in directories[0]:
        test_set_name = "big5_100"
    else:
        test_set_name = "pvq_male"


    dir_2_data = {}
    for i, directory in enumerate(directories):
        if not os.path.isdir(directory):
            continue
        results_json_path = os.path.join(directory, 'results.json')
        if not os.path.isfile(results_json_path):
            continue

        with open(results_json_path, 'r') as f:
            data = json.load(f)
        dir_2_data[directory] = data


    if test_set_name == "pvq_male":
        test_set_values = [
            'Conformity',
            'Tradition',
            'Benevolence',
            'Universalism',
            'Self-Direction',
            'Stimulation',
            'Hedonism',
            'Achievement',
            'Power',
            'Security'
        ]
    elif test_set_name == "hofstede":
        test_set_values = [
            "Power Distance",
            "Individualism",
            "Masculinity",
            "Uncertainty Avoidance",
            "Long-Term Orientation",
            "Indulgence",
        ]
    elif test_set_name in ["big5_50", "big5_100"]:
        test_set_values = [
            "Neuroticism",
            "Extraversion",
            "Openness to Experience",
            "Agreeableness",
            "Conscientiousness"
        ]

    primary_value_alignments = []
    mean_vars = []
    labels = []
    if all(["Primary Values".lower() in d.lower() for d in directories]) or True:
        for dir_i, (dir, data) in enumerate(dir_2_data.items()):

            labels.append(dir_to_label(dir))

            normalized_evaluation_data.append([])
            notnorm_evaluation_data.append([])

            if all(["Primary Values".lower() in d.lower() for d in directories]):
                profile = {}
                # extract values from profile string
                if "params" in data:
                    profile_str = data['params']['profile']
                else:
                    profile_str = dir[dir.rindex("profile"):dir.index("_2023")]

                for item in profile_str.split(';'):
                    key, value = item.split(':')
                    profile[key] = value

                primary_values = profile["Primary values"].split(",")

                if "Primary values" not in profile:
                    raise ValueError(f"Primary values are not in the profile: {profile}.")

                assert all([prim_v in test_set_values for prim_v in primary_values])
            else:
                primary_values = None

            from collections import defaultdict
            normalizing_constants = {
                    "pvq_male": defaultdict(lambda: 5),
                    "hofstede": {
                        "Power Distance": 2*300,
                        "Individualism": 2*350,
                        "Masculinity": 2*350,
                        "Uncertainty Avoidance": 2*325,
                        "Long-Term Orientation": 2*325,
                        "Indulgence": 2*375,
                    },
                    "big5_50": defaultdict(lambda: 50),
            }

            normalizing_offset = {
                "pvq_male": defaultdict(lambda: -1),
                "hofstede": {
                    "Power Distance": 300,
                    "Individualism": 350,
                    "Masculinity": 350,
                    "Uncertainty Avoidance": 325,
                    "Long-Term Orientation": 325,
                    "Indulgence": 375,
                },
                "big5_50": defaultdict(lambda: 0),
            }

            per_value_norm_scores = defaultdict(list)

            for perm_i, perm_metrics in enumerate(data["per_permutation_metrics"]):

                # we normalize by (x+normalizing_offset)/normalizing_constant
                notnorm_perm_metrics = {val: perm_metrics[test_set_name][val] for val in test_set_values}

                norm_perm_metrics = {
                    val: (
                                 perm_metrics[test_set_name][val]+normalizing_offset[test_set_name][val]
                    )/normalizing_constants[test_set_name][val] for val in test_set_values
                }

                for val, score in norm_perm_metrics.items():
                    per_value_norm_scores[val].append(score)

                if primary_values:
                    avg_primary_values = np.mean([norm_perm_metrics[val] for val in primary_values])
                    avg_other_values = np.mean(
                        [norm_perm_metrics[val] for val in list(set(test_set_values) - set(primary_values))])
                    perm_alignment = avg_primary_values - avg_other_values
                    # print("permutation alignment: ", perm_alignment)
                    primary_value_alignments.append(perm_alignment)

                normalized_evaluation_data[dir_i].append([norm_perm_metrics[v] for v in test_set_values])
                notnorm_evaluation_data[dir_i].append([notnorm_perm_metrics[v] for v in test_set_values])


            # mean (over values/traits) of vars (over permutations)
            mean_vars.append(np.mean([np.var(norm_scores) for norm_scores in per_value_norm_scores.values()]))

            if primary_values:
                perspective_value_alignment = np.mean(primary_value_alignments[-len(data["per_permutation_metrics"]):])

                # dump alignemens to json

                dump_json_path = dir + "/alignments.json"
                with open(dump_json_path, 'w') as f:
                    json.dump(primary_value_alignments[-len(data["per_permutation_metrics"]):], f)

                # print(f"Primary value alignment for {primary_values}: {perspective_value_alignment}.")

        if primary_values:
            # todo: confirm this
            mean_primary_value_alignment = np.mean(primary_value_alignments)
            print(colored(f"Mean primary value alignment (over all): {round(mean_primary_value_alignment, 3)}", "green"))

        # mean over perspectives
        mean_var = np.mean(mean_vars)

    # all_evaluation_data = (n_persp, n_perm, n_values/traits)
    normalized_evaluation_data = np.array(normalized_evaluation_data)
    notnorm_evaluation_data = np.array(notnorm_evaluation_data)

    # mean over value dimensions
    # mean_variance = variances.mean()

    # print(f"Mean (over values) Variance (over perspectives): {mean_variance}")
    perm_var = normalized_evaluation_data.var(1).mean()
    # var(permut) -> mean (persp, values)
    assert np.isclose(perm_var, mean_var)
    print(f"Permutation Var - mean (over values/traits x perspectives) of var (over perm) (*10^3): {round(perm_var * (10 ** 3), 2)}")

    persp_var = normalized_evaluation_data.mean(1).var(0).mean()
    # mean(permut) -> var (persp) -> mean (values)
    print(f"Perspective Var - mean (over values/traits) of var (over perspectives) of mean (over perm) (*10^3): {round(persp_var * (10 ** 3), 2)}")

    # all_evaluation_data = (n_perspectives, n_permutations, n_values/traits)
    # we do a separate anova for each value/trait
    assert normalized_evaluation_data.shape[-1] == len(test_set_values)

    # p limit for the group null-hyp - all perspectivea are the same distribution

    p_limit = 0.05
    # p_limit = 0.005
    # p_limit = 0.001

    # group_p_limit = 0.001
    group_p_limit = p_limit / len(test_set_values)  # additional bonferroni correction

    print("testing p-value: ", group_p_limit)

    if test_set_name == "pvq_male":
        # PVQ centering - is this reccomented in this manual? # todo: do properly all questions
        notnorm_evaluation_data = notnorm_evaluation_data - np.repeat(notnorm_evaluation_data.mean(2)[:,:, np.newaxis], 10, axis=2)

    for val_i, val in enumerate(test_set_values):
        # value_data = normalized_evaluation_data[:, :, val_i] # value_data = (n_perspectives, n_permutations)
        value_data = notnorm_evaluation_data[:, :, val_i] # value_data = (n_perspectives, n_permutations)

        print(f"\n----------------{val}---------------------")

        assume_assumptions=True

        if assume_assumptions:
            # it's usually ok to violate assumptions
            levene_ok = True
            gaussian = True

        else:
            # for assumptions
            alpha = 0.001

            # LEVENE test for equal variance (assumption for anova and tukey)
            levene_test_stat, levene_p_value = stats.levene(*value_data)
            if levene_p_value < alpha:  # or whatever alpha level you choose
                levene_ok = False
                print(f"Levene not OK (p={levene_p_value}) - not same variances")

            else:
                levene_ok = True
                print(f"Levene OK (p={levene_p_value}) - same variances")

            # Shapiro - normality
            gaussian = []
            ps = []
            for v_d in value_data:
                _, p = shapiro(v_d)
                ps.append(p)

                if p < alpha:
                    gaussian.append(False)
                else:
                    gaussian.append(True)

                # plt.hist(v_d)
                # plt.title(f"p = {p} - {'' if gaussian[-1] else 'not'} gaussian")
                # plt.show()

            gaussian = all(gaussian)
            # print(f"Gaussian (p={ps}): {gaussian}")
            print(f"Gaussian (Shapiro) (p={np.round(ps, 5)}): {gaussian}")

        if gaussian:
            if levene_ok:
                # standard ANOVA
                testname = "ANOVA"
                _, pvalue = stats.f_oneway(*value_data)
            else:
                # Welch ANOVA
                test_name = "Welch ANOVA"
                df = pd.DataFrame({'score': np.array(value_data).flatten(),
                                   'group': np.repeat(labels, repeats=50)})

                p_value = pg.welch_anova(dv='score', between='group', data=df)["p-unc"].values
                assert len(p_value) == 1
                p_value = p_value[0]

        else:
            # kruskal
            testname = "Krushal-Wallis (H-test)"
            _, pvalue = stats.kruskal(*value_data)


        if pvalue >= group_p_limit:
            different_distr.append(False)
            print(colored(f"p({testname}): {pvalue} > {group_p_limit}", "red"))
        else:
            different_distr.append(True)
            print(colored(f"p({testname}): {pvalue} < {group_p_limit}", "green"))

            pairs_ind = list(itertools.combinations(range(len(value_data)), 2))

            if gaussian and levene_ok:
                # Tukey
                tukey_res=tukey_hsd(*value_data)

                # print("\tTukey (p limit) :", p_limit)
                print("\tTukey (p limit) :", group_p_limit)
                for pair_ind in pairs_ind:

                    res_pvalue = tukey_res.pvalue[pair_ind[0], pair_ind[1]]

                    if res_pvalue < p_limit:
                        d = cohen_d(value_data[pair_ind[0]], value_data[pair_ind[1]])
                        print("\t{} - {} : p -> {:.8f} d -> {:.8f}".format(labels[pair_ind[0]], labels[pair_ind[1]], res_pvalue, d))

            else:
                # Dunn
                dunn_res = sp.posthoc_dunn(value_data, p_adjust='bonferroni')
                for pair_ind in pairs_ind:
                    res_pvalue = dunn_res.iloc[pair_ind[0], pair_ind[1]]
                    if res_pvalue < p_limit:
                        print("\t{} - {} : {}".format(labels[pair_ind[0]], labels[pair_ind[1]], res_pvalue))


            # # bonferroni correction
            # n_comparisons = len(pairs_ind)
            # bonf_p_limit = p_limit / len(pairs_ind)
            #
            # print(f"\tBonferroni corrected p : {p_limit} -> {bonf_p_limit}")
            # print(f"\t{'t-test' if gaussian else 'u-test'} p < {bonf_p_limit} for: ")
            #
            # for pair_ind in pairs_ind:
            #     if gaussian:
            #         # welch or standard t-test (welch - leven_ok - equal_var:True)
            #         res_pvalue = stats.ttest_ind(a=value_data[pair_ind[0]], b=value_data[pair_ind[1]], equal_var=levene_ok).pvalue
            #     else:
            #         from IPython import embed; embed();
            #
            #         # Mann-Whitney U
            #         res_pvalue = stats.mannwhitneyu(x=value_data[pair_ind[0]], y=value_data[pair_ind[1]]).pvalue
            #
            #     pair_labels = (labels[pair_ind[0]], labels[pair_ind[1]])
            #     if res_pvalue < bonf_p_limit:
            #         print("\t{} -> {} < {}".format(pair_labels, res_pvalue, bonf_p_limit))


    # plot bars
    for distr_i, different in enumerate(different_distr):
        if not different:
            # anova didn't reject - same distr
            plt.bar(distr_i, 300, width=all_bars_width*1.3, color="lightgray")
            plt.bar(distr_i, -300, width=all_bars_width*1.3, color="lightgray")

    orders = {}
    orders_kv = {}
    for i, directory in enumerate(directories):

        # from IPython import embed; embed();
        if not os.path.isdir(directory):
            continue

        results_json_path = os.path.join(directory, 'results.json')
        if not os.path.isfile(results_json_path):
            continue

        with open(results_json_path, 'r') as f:
            data = json.load(f)


        offset = -all_bars_width/2 + (i/num_dirs)*all_bars_width
        keys_, sorted_keys, keys_vals = plot_baseline(data, ax, directory, offset, keys_to_plot=keys_to_plot, bar_width=bar_width, min_bar_size=0.05, horizontal_bar=args.horizontal_bar, all_evaluation_data=normalized_evaluation_data)
        orders[dir_to_label(directory)] = sorted_keys
        orders_kv[dir_to_label(directory)] = keys_vals
        assert keys_ == test_set_values

        # check that keys are the same in all the baselines
        assert keys is None or keys_ == keys
        keys = keys_


    print()

    ########################
    # CORRELATIONS
    ########################

    print(colored("Pearson Correlation (between perspectives) - permutation order", "green"))
    corrs = defaultdict(list)

    perm_orders = defaultdict(dict)

    for dir_1, dir_2 in combinations(directories, 2):

        lab_1 = dir_to_label(dir_1)
        lab_2 = dir_to_label(dir_2)

        print(f"{lab_1} vs {lab_2}")
        for key in keys:
            # perm -> score
            perm_i_score_1 = {i: d[test_set_name][key] for i, d in list(enumerate(dir_2_data[dir_1]["per_permutation_metrics"]))}
            perm_i_score_2 = {i: d[test_set_name][key] for i, d in list(enumerate(dir_2_data[dir_2]["per_permutation_metrics"]))}

            # save permutation orderings
            # trait/value -> dict -> perm index in order of score
            perm_orders[key][dir_1] = [k for k, v in sorted(perm_i_score_1.items(), key=lambda item: item[1])]
            perm_orders[key][dir_2] = [k for k, v in sorted(perm_i_score_2.items(), key=lambda item: item[1])]

            if compare_scores:
                # perms scores in order of scores
                values1 = [perm_i_score_1[k] for k in perm_orders[key][dir_1]]
                assert is_strictly_increasing(values1)  # scores are increasing here

                # same order of permutations with other scores
                values2 = [perm_i_score_2[k] for k in perm_orders[key][dir_1]]

            else:
                # compare ranks
                mapping = {v: i for (i, v) in enumerate(perm_orders[key][dir_1])}
                values1 = [mapping[v] for v in perm_orders[key][dir_1]]
                values2 = [mapping[v] for v in perm_orders[key][dir_2]]
                assert values1 == list(range(len(values1)))


            if spearman:
                correlation, _ = spearmanr(values1, values2)
            else:
                correlation, _ = pearsonr(values1, values2)

            if np.isnan(correlation):
                assert compare_scores
                # Constant values in one group. Correlation is undefined and artificially set to 0
                correlation = 0.0

            corrs[key].append(correlation)

            print(f"\t{key} : {correlation}")

    for key in keys:
        print(f"{key}:")
        print("\tAvg Pearson corr:", np.mean(corrs[key]))
        print("\tMin Pearson corr:", np.min(corrs[key]))

    all_corrs = list(itertools.chain(*corrs.values()))
    print_aggregated_correlation_stats(all_corrs)

    print("--------------------------------------------------")

    # order of perspectives
    print(colored("Pearson Correlation (between permutations) - perspective order", "green"))
    corrs = defaultdict(list)
    persp_orders = defaultdict(dict)

    n_perms = len(dir_2_data[dir_1]["per_permutation_metrics"])

    assert n_perms == 50
    assert n_perms == len(dir_2_data[dir_2]["per_permutation_metrics"])

    for perm_1, perm_2 in combinations(range(n_perms), 2):

        for key in keys:
            persp_score_1 = {d: dir_2_data[d]["per_permutation_metrics"][perm_1][test_set_name][key] for d in directories}
            persp_score_2 = {d: dir_2_data[d]["per_permutation_metrics"][perm_2][test_set_name][key] for d in directories}

            # save key orderings
            persp_orders[key][perm_1] = [k for k, v in sorted(persp_score_1.items(), key=lambda item: item[1])]
            persp_orders[key][perm_2] = [k for k, v in sorted(persp_score_2.items(), key=lambda item: item[1])]


            if compare_scores:
                # compare scores
                values1 = [persp_score_1[k] for k in persp_orders[key][perm_1]]
                assert is_strictly_increasing(values1)  # scores are increasing here

                # same order of perspectives with other scores
                values2 = [persp_score_2[k] for k in persp_orders[key][perm_1]]

            else:
                # compare ranks
                mapping = {v: i for (i, v) in enumerate(persp_orders[key][perm_1])}
                values1 = [mapping[v] for v in persp_orders[key][perm_1]]
                values2 = [mapping[v] for v in persp_orders[key][perm_2]]

            if spearman:
                correlation, _ = spearmanr(values1, values2)
            else:
                correlation, _ = pearsonr(values1, values2)

            if np.isnan(correlation):
                assert compare_scores
                # Constant values in one group. Correlation is undefined and artificially set to 0
                correlation = 0.0

            corrs[key].append(correlation)

    for key in keys:
        print(key)
        print("\tAvg Pearson corr:", np.mean(corrs[key]))
        print("\tMin Pearson corr:", np.min(corrs[key]))

    all_corrs = list(itertools.chain(*corrs.values()))

    print_aggregated_correlation_stats(all_corrs)

    print("--------------------------------------------------")


    # order of perspectives - avg over permutations first (average scores)
    print(colored("Pearson Correlation (between perspectives) - values order (average scores)", "green"))
    corrs = list()
    key_orders = defaultdict(dict)
    persp_scores = defaultdict(dict)

    n_perms = len(dir_2_data[dir_1]["per_permutation_metrics"])

    assert n_perms == 50
    assert n_perms == len(dir_2_data[dir_2]["per_permutation_metrics"])

    for d in directories:
        for key in keys:
            persp_scores[d][key] = np.mean([dir_2_data[d]["per_permutation_metrics"][i][test_set_name][key] for i in range(50)])

        # keys in order
        key_orders[d] = [k for k, v in sorted(persp_scores[d].items(), key=lambda item: item[1])]

    for dir1, dir2 in combinations(directories, 2):

        if compare_scores:
            # key scores in order
            values1 = [persp_scores[dir1][k] for k in key_orders[dir1]]
            assert is_strictly_increasing(values1)
            # same order of keys (values/traits) with other scores
            values2 = [persp_scores[dir2][k] for k in key_orders[dir1]]

        else:
            mapping = {v: i for (i, v) in enumerate(key_orders[dir1])}

            values1 = [mapping[v] for v in key_orders[dir1]]
            values2 = [mapping[v] for v in key_orders[dir2]]

        if spearman:
            correlation, _ = spearmanr(values1, values2)
        else:
            correlation, _ = pearsonr(values1, values2)

        if np.isnan(correlation):
            assert compare_scores
            # Constant values in one group. Correlation is undefined and artificially set to 0
            correlation = 0.0

        print(f"\t{dir_to_label(dir1)} - {dir_to_label(dir2)} : {correlation}")
        corrs.append(correlation)

    print_aggregated_correlation_stats(corrs)


    print("--------------------------------------------------")

    # order of perspectives - avg over permutations last (average r)
    print(colored("Pearson Correlation (between perspectives) - values order (average r)", "green"))
    corrs = list()
    key_orders = defaultdict(dict)
    persp_scores = defaultdict(lambda : defaultdict(dict))

    n_perms = len(dir_2_data[dir_1]["per_permutation_metrics"])

    assert n_perms == 50
    assert n_perms == len(dir_2_data[dir_2]["per_permutation_metrics"])

    for p_i in range(50):
        for d in directories:
            for key in keys:
                persp_scores[p_i][d][key] = dir_2_data[d]["per_permutation_metrics"][p_i][test_set_name][key]

            # keys in order
            key_orders[p_i][d] = [k for k, v in sorted(persp_scores[p_i][d].items(), key=lambda item: item[1])]

    for dir1, dir2 in combinations(directories, 2):
        correlation_list = []

        for p_i in range(50):

            if compare_scores:
                # compare scores
                # key (traits/values) scores in order
                values1 = [persp_scores[p_i][dir1][k] for k in key_orders[p_i][dir1]]
                assert is_strictly_increasing(values1)

                # same order of keys (traits/values) with other scores
                values2 = [persp_scores[p_i][dir2][k] for k in key_orders[p_i][dir1]]

            else:
                # compare ranks
                mapping = {v: i for (i, v) in enumerate(key_orders[p_i][dir1])}
                values1 = [mapping[v] for v in key_orders[p_i][dir1]]
                values2 = [mapping[v] for v in key_orders[p_i][dir2]]

            if spearman:
                correlation, _ = spearmanr(values1, values2)
            else:
                correlation, _ = pearsonr(values1, values2)

            if np.isnan(correlation):
                assert compare_scores
                # Constant values in one group. Correlation is undefined and artificially set to 0
                correlation = 0.0

            correlation_list.append(correlation)

        correlation = np.mean(correlation_list) # average correlation in value order

        print(f"{dir_to_label(dir1)} - {dir_to_label(dir2)}:")
        print_correlation_stats(correlation_list)

        corrs.append(correlation)

    print_aggregated_correlation_stats(corrs)



    if args.horizontal_bar:

        # Set the y-ticks labels on the left side
        y_locs = list(range(len(keys)))

        # right labels
        key_to_hofstede_label_right = {
            "power_distance": "high power distance",
            "individualism": "individualistic",
            "masculinity": "masculine",
            "uncertainty_avoidance": "high uncertainty_avoidance",
            "long_term_orientation": "long term orientation",
            "indulgence": "indulgence",
        }

        key_to_hofstede_label_left = {
            "power_distance": "low power distance",
            "individualism": "collectivistic",
            "masculinity": "feminine",
            "uncertainty_avoidance": "low uncertainty avoidance",
            "long_term_orientation": "short term orientation",
            "indulgence": "restraint",
        }

        left_labels = [key_to_hofstede_label_left.get(k) for k in keys]
        right_labels = [key_to_hofstede_label_right.get(k) for k in keys]

        # add left ticks
        ax.set_yticks(y_locs)
        ax.set_yticklabels(left_labels)

        ax2 = ax.twinx()
        ax2.set_ylim(ax.get_ylim())
        ax2.set_yticks(y_locs)
        ax2.set_yticklabels(right_labels)

    else:
        ax.set_xticks(range(len(keys)))

        if test_set_name == "pvq_male":
            ax.set_xticklabels(keys, rotation=90)
        else:
            ax.set_xticklabels(keys, rotation=90)

        ax.tick_params(axis='both', which='both', labelsize=30)


    if figure_draw:
        ax.legend().remove()
        ax.set_title("")

    if args.save:
        # for ext in ["png", "svg"]:
        for ext in ["png"]:
            savepath = f"visualizations/{args.filename}.{ext}"
            print(f"Saved to: {savepath}")
            # plt.tight_layout()
            plt.savefig(savepath)

            if args.separate_legend:
                # create a new figure and axes object for the legend
                fig_legend, ax_legend = plt.subplots(figsize=(4, 4))

                # call the legend() method with the original axes object and the new axes object
                ax_legend.legend(*ax.get_legend_handles_labels(), loc="center")

                # remove the x and y ticks from the legend axes object
                ax_legend.set_xticks([])
                ax_legend.set_yticks([])

                # ax.set_frame_on(False)
                # ax.xaxis.set_visible(False)
                # ax.yaxis.set_visible(False)
                # ax.axis('off')

                # save the legend as a separate image
                savepath_legend = f"visualizations/{args.filename}_legend.{ext}"
                print(f"Saved legend to: {savepath_legend}")
                plt.tight_layout()
                plt.gca().set_axis_off()
                plt.savefig(savepath_legend)

    else:
        plt.show()

