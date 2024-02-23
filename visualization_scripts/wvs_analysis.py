import os
import json
import warnings
import random

import matplotlib.pyplot as plt
import re
import numpy as np
from termcolor import colored
import scipy.stats as stats
import itertools
from scipy.stats import tukey_hsd, sem, rankdata
from scipy.stats import pearsonr, spearmanr, ConstantInputWarning
from collections import defaultdict

from scipy.spatial import distance


def extract_test_set_name(dir_2_data):
    test_set_names = set(itertools.chain(*[v['metrics'].keys() for v in dir_2_data.values()]))

    assert len(test_set_names) == 1
    test_set_name = list(test_set_names)[0]
    return test_set_name


def extract_test_set_values(dir_2_data):
    test_set_values_ = [list(list(v['metrics'].values())[0].keys()) for v in dir_2_data.values()]
    # test_set_values_ = [list(v['metrics'][test_set_name].keys()) for v in dir_2_data.values()]
    assert len(set([t.__repr__() for t in test_set_values_])) == 1
    test_set_values = test_set_values_[0]

    return test_set_values


def load_data(directories):
    # load data
    data = {}
    for i, directory in enumerate(directories):
        if not os.path.isdir(directory):
            continue

        results_json_path = os.path.join(directory, 'results.json')

        if not os.path.isfile(results_json_path):
            continue

        with open(results_json_path, 'r') as f:
            dir_data = json.load(f)

        test_name = extract_test_set_name({directory: dir_data})

        # parse tolkien

        # separated fair unfair races by default
        parse_metrics = True  # better with fair/unfair
        parse_good_bad = False  # better without

        if parse_metrics:

            def parse_dir(d):


                if "tolkien_donation" in directory:
                    return d

                elif "pvq" in directory:
                    return d

                else:
                    return d

            dir_data['metrics'][test_name] = parse_dir(dir_data['metrics'][test_name])

            dir_data['per_permutation_metrics'] = [
                {test_name: parse_dir(d[test_name])} for d in dir_data['per_permutation_metrics']
            ]
            dir_data['per_simulated_participant_metrics'] = [
                {test_name: parse_dir(d[test_name])} for d in dir_data['per_simulated_participant_metrics']
            ]

        data[directory] = dir_data

    return data


def is_strictly_increasing(lst):
    return all(x <= y for x, y in zip(lst, lst[1:]))


def print_dict_values(d):
    print("\t".join([f"{k:<10}" for k in list(d.keys()) + ["Mean"]]))
    print("\t\t".join([f"{np.round(s, 2):.2}" for s in list(d.values()) + [np.mean(list(d.values()))]]))



def dir_to_label(directory):

    if "format_chat_simulate_conv" in directory:
        label = extract_value(directory, "_simulate_conv_")

    elif "simulate_conv" in directory:
        label = extract_value(directory, "_simulate_conv_")

    elif "weather" in directory:
        label = extract_value(directory, "_weather_")

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


def extract_value(directory, key="_lotr_character_"):
    label = os.path.basename(directory)
    if key in label:
        start_index = label.find(key) + len(key)

    elif "_ntrain_" in label:
        start_index = label.find("ntrain_") + len("ntrain_") + 1

    else:
        start_index = 0

    match = re.search(r"(_+202\d)", label)

    if match:
        end_index = label.find(match.group(0))
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



if __name__ == '__main__':
    import argparse

    # normalized_evaluation_data = []
    # notnorm_evaluation_data = []
    vals = []

    parser = argparse.ArgumentParser()
    parser.add_argument('directories', nargs='+', help='directories containing results.json files')
    parser.add_argument('--plot-save', '-ps', action="store_true")
    parser.add_argument('--insert-dummy', '-id', action="store_true")
    parser.add_argument('--separate_legend', action="store_true")
    parser.add_argument('--plot-ranks', "-pr", action="store_true")
    parser.add_argument('--plot-ips', "-pi", action="store_true")
    parser.add_argument('--plot-mean', "-pm", action="store_true")
    parser.add_argument('--plot-dont-show', "-pds", action="store_true")
    parser.add_argument('--filename', type=str, default="hobbies_pvq")
    parser.add_argument('--assert-n-dirs', type=int, default=None)
    parser.add_argument('--result-json-stdout', action="store_true")
    args = parser.parse_args()

    if args.result_json_stdout:
        # redurect stdout to null
        import sys
        sys.stdout = open(os.devnull, 'w')

    different_distr = [] # value/traits where the anova test said it's different

    keys_to_plot = None

    bar_width = 0.10
    bar_margin = 1.2

    mean_primary_value_alignment = None
    spearman = True
    if spearman:
        print("Spearman")

    if args.insert_dummy:
        print("Inserting dummy participants.")

    if args.plot_mean:
        fig, ax = plt.subplots(figsize=(15, 10))

    ignore = [
        "religion",
        "tax",
        "vacation",
        # "format_chat___",
        # "grammar",
        # "poem",
        # "joke",
        # "history",
        # "chess",
    ]

    must_have = ["format_chat___"]
    must_have = ["_"]

    args.directories = [d for d in args.directories if not any([i in d for i in ignore])]
    args.directories = [
        d for d in args.directories if any([m in d for m in must_have])
    ]

    num_dirs = len([d for d in args.directories if os.path.isdir(d)])
    all_bars_width = num_dirs * (bar_width*bar_margin)  # bars with margins

    # chronological order
    directories = args.directories

    # remove directories which contain substrings from the list
    ignore_patterns = []
    print("Ignoring patterns: ", ignore_patterns)

    for substring in ignore_patterns:
        directories = [d for d in directories if substring not in d]

    directories = [d for d in directories if os.path.isfile(os.path.join(d, 'results.json'))]

    print("Directories:\n\t", "\n\t".join(directories))

    if len(directories) < 2:
        raise IOError(f"Only {len(directories)} result.json files found.")

    if args.assert_n_dirs and (len(directories) != args.assert_n_dirs):
        raise ValueError(f"Wrong number of dirs found {len(directories)} != {args.assert_n_dirs}.")

    dir_2_data = load_data(directories)

    test_set_name = extract_test_set_name(dir_2_data)
    test_set_values = extract_test_set_values(dir_2_data)

    pop_1 = "elves"
    pop_2 = "humans"
    tot_sim = []
    for dir in directories:
        histrs_1 = dir_2_data[dir]['pop_metrics'][pop_1]['hist']
        histrs_2 = dir_2_data[dir]['pop_metrics'][pop_2]['hist']

        pair_sim = []
        for histr_1, histr_2 in zip(histrs_1, histrs_2):
            max_key = max(map(int, itertools.chain(histr_1.keys(), histr_2.keys())))
            keys = list(map(str, range(1, max_key + 1)))

            N = sum(histr_1.values())
            probs_1 = [float(histr_1.get(k, 0)) / N for k in keys]
            probs_2 = [float(histr_2.get(k, 0)) / N for k in keys]

            sim_q = 1 - distance.jensenshannon(probs_1, probs_2)
            pair_sim.append(sim_q)

        # print(f"----------------------\n{dir_1}\n{dir_2}\n---> {np.mean(pair_sim)}")
        tot_sim.append(pair_sim)

    final_sim = np.mean(tot_sim)
    print(f"Similarity {pop_1} - {pop_2}:", final_sim)

    pop = "elves"
    tot_sim = []

    for dir_1, dir_2 in itertools.combinations(directories, 2):
        histrs_1 = dir_2_data[dir_1]['pop_metrics'][pop]['hist']
        histrs_2 = dir_2_data[dir_2]['pop_metrics'][pop]['hist']

        pair_sim = []
        for histr_1, histr_2 in zip(histrs_1, histrs_2):
            max_key = max(map(int, itertools.chain(histr_1.keys(), histr_2.keys())))
            keys = list(map(str,range(1, max_key+1)))

            N = sum(histr_1.values())
            probs_1 = [float(histr_1.get(k, 0))/N for k in keys]
            probs_2 = [float(histr_2.get(k, 0))/N for k in keys]

            sim_q = 1 - distance.jensenshannon(probs_1, probs_2)
            pair_sim.append(sim_q)

        # print(f"----------------------\n{dir_1}\n{dir_2}\n---> {np.mean(pair_sim)}")
        tot_sim.append(pair_sim)

    final_sim = np.mean(tot_sim)
    print(f"Similarity {pop} (contexts):", final_sim)

    if args.result_json_stdout:
        sys.stdout = sys.__stdout__

        outputs = {"Similarity": final_sim}

        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return json.JSONEncoder.default(self, obj)

        print(json.dumps(outputs, cls=NumpyEncoder))

