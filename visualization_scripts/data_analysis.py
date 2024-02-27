import copy
import os
import json
import warnings
import random
import glob

import matplotlib.pyplot as plt
import re
import numpy as np
from termcolor import colored
import scipy.stats as stats
import itertools
from scipy.stats import tukey_hsd, sem, rankdata
from scipy.stats import pearsonr, spearmanr, ConstantInputWarning
from collections import defaultdict


def per_part_normalized_scores(dir_2_data, dir, test_set_name, key):
    assert "pvq" in test_set_name
    scores = np.array([d[test_set_name][key] for d in dir_2_data[dir]["per_simulated_participant_metrics"]])
    average_part_answer = np.array([np.array(d[test_set_name])[:, 1].astype(float).mean() for d in dir_2_data[dir]["answers"]])
    scores -= average_part_answer
    return scores


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

        if parse_metrics:

            def parse_dir(d):

                if "tolkien_donation" in directory:
                    # return {"Donated": np.mean(list(map(float, d.values())))}
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


def compute_ipsative_stability(dir_2_data, keys, default_profile=None):

    # IPSATIVE CORRELATIONS
    print(colored("\n\n--------------------------------------------------", "green"))
    print(colored(f"IPSATIVE CORELATIONS {'WITH DEFAULT PROFILE' if default_profile is not None else ''}", "green"))
    print(colored("--------------------------------------------------", "green"))

    # order of contexts - avg over permutations last (average r)
    print("Pearson Correlation (between contexts) - values order (average r)")
    key_orders = defaultdict(dict)
    part_scores = defaultdict(lambda: defaultdict(dict))

    directories = list(dir_2_data.keys())

    population = dir_2_data[directories[0]]['simulated_population']

    if set(population) == {None}:
        population = list(range(len(population)))

    for p_i, part in enumerate(population):
        for d in directories:
            for key in keys:
                scores = dir_2_data[d]["per_permutation_metrics"][p_i][test_set_name][key]

                if "pvq" in test_set_name:
                    # todo: remove
                    average_part_answer = np.array(dir_2_data[d]["answers"][p_i][test_set_name])[:, 1].astype(float).mean()
                    scores_ = scores - average_part_answer
                    scores = per_part_normalized_scores(dir_2_data, d, test_set_name, key)[p_i]
                    assert scores == scores_

                part_scores[part][d][key] = scores

            # keys in order
            key_orders[part][d] = [k for k, v in sorted(part_scores[part][d].items(), key=lambda item: item[1])]

    headers = "\t".join(["Mean", "Median", "STD", "Min", "Max"])
    print("".ljust(20, " ") + f"{headers}")

    all_corrs = list()

    dir_stabilities = defaultdict(lambda : defaultdict(list))
    ips_part_stabilities = defaultdict(list)

    if default_profile is not None:
        for dir_ in directories:
            population_correlations = []

            for part in population:
                scores_ = [part_scores[part][dir_][k] for k in keys]
                corr = compute_correlation(scores_, default_profile, spearman, insert_dummy=args.insert_dummy)

                dir_stabilities[part][dir_].append(corr)

                ips_part_stabilities[part].append(corr)

                population_correlations.append(corr)

            label = f"{dir_to_label(dir_)}:"
            print_correlation_stats(population_correlations, label=label)

            all_corrs.append(population_correlations)

    else:
        for dir1, dir2 in itertools.combinations(directories, 2):
            population_correlations = []

            for part in population:

                scores_1 = [part_scores[part][dir1][k] for k in keys]
                scores_2 = [part_scores[part][dir2][k] for k in keys]

                corr = compute_correlation(scores_1, scores_2, spearman, insert_dummy=args.insert_dummy)

                dir_stabilities[part][dir1].append(corr)
                dir_stabilities[part][dir2].append(corr)

                ips_part_stabilities[part].append(corr)

                population_correlations.append(corr)

            label = f"{dir_to_label(dir1)} - {dir_to_label(dir2)}:"
            print_correlation_stats(population_correlations, label=label)

            all_corrs.append(population_correlations)

    all_corrs = np.array(all_corrs) # n_comps x n_pop

    ips_part_stabilities = {p: np.mean(v) for p, v in ips_part_stabilities.items()}
    ips_part_dir_stabilities = {p: {d: np.mean(v_) for d, v_ in v.items()} for p, v in dir_stabilities.items()}

    # mean over participants first just for display
    print_correlation_stats(all_corrs.mean(axis=1), label="Mean", color="blue")

    mean_ipsative_stability = all_corrs.mean()

    return mean_ipsative_stability, part_scores, ips_part_stabilities, ips_part_dir_stabilities, all_corrs

def compute_paired_rank_order_stability(dir_2_data_1, dir_2_data_2, key_1, key_2, test_set_name_1, test_set_name_2, verbose=False, directories_1=None, directories_2=None):

    if verbose:
        print(colored("\n\n--------------------------------------------------", "green"))
        print(colored("PAIRED RANK ORDER STABILITY", "green"))
        print(colored("--------------------------------------------------", "green"))

        print("Rank Order stability - Pearson Correlation in simulated participant order (between two contexts)\n")

    corrs = defaultdict(list)

    if directories_1 is None or directories_2 is None:
        directories_1, directories_2 = list(dir_2_data_1.keys()), list(dir_2_data_2.keys())

    dir_pairs = zip(directories_1, directories_2)
    dir_2_data = {**dir_2_data_1, **dir_2_data_2}

    key_dir_corrs = defaultdict(lambda: defaultdict(list))

    for dir_1, dir_2 in dir_pairs:
        lab_1, lab_2 = dir_to_label(dir_1), dir_to_label(dir_2)

        scores_1 = np.array([d[test_set_name_1][key_1] for d in dir_2_data[dir_1]["per_simulated_participant_metrics"]])
        scores_2 = np.array([d[test_set_name_2][key_2] for d in dir_2_data[dir_2]["per_simulated_participant_metrics"]])

        if "pvq" in test_set_name_1:
            scores_1 = per_part_normalized_scores(dir_2_data, dir_1, test_set_name_1, key_1)

        if "pvq" in test_set_name_2:
            scores_2 = per_part_normalized_scores(dir_2_data, dir_2, test_set_name_2, key_2)

        correlation = compute_correlation(scores_1, scores_2, spearman, insert_dummy=args.insert_dummy)

        if verbose:
            print(f"{lab_1} - {lab_2} : {correlation}")

        corrs[key_1].append(correlation)

        key_dir_corrs[key_1][dir_1].append(correlation)
        key_dir_corrs[key_1][dir_2].append(correlation)

    key_rank_order_stabilities = {key_1: np.mean(corrs[key_1])}
    mean_stabilities = list(key_rank_order_stabilities.values())

    mean_rank_order_stability = np.mean(mean_stabilities)

    all_corrs = list(itertools.chain(*corrs.values()))

    if verbose:
        print(f"\nAverage over context changes ({len(directories)}) and values ({len(keys)})")
        print_aggregated_correlation_stats(all_corrs)

    key_dir_corrs = {
        k: {
            d: np.mean(v_) for d, v_ in v.items()
        } for k, v in key_dir_corrs.items()}

    return mean_rank_order_stability, key_rank_order_stabilities, key_dir_corrs

def compute_rank_order_stability(dir_2_data, keys):

    print(colored("\n\n--------------------------------------------------", "green"))
    print(colored("RANK ORDER STABILITY", "green"))
    print(colored("--------------------------------------------------", "green"))

    print("Rank Order stability - Pearson Correlation in simulated participant order (between two contexts)\n")
    corrs = defaultdict(list)

    directories = list(dir_2_data.keys())


    print(" ".ljust(35, ' ') + "& " + " & ".join(keys + ["Mean"]) + " \\\\")

    key_dir_corrs = defaultdict(lambda: defaultdict(list))

    dir_pairs = itertools.combinations(directories, 2)

    for dir_1, dir_2 in dir_pairs:
        lab_1 = dir_to_label(dir_1)
        lab_2 = dir_to_label(dir_2)
        # print(f"{lab_1} vs {lab_2}:")
        lab_latex = f"{lab_1} - {lab_2}"

        for key in keys:

            scores_1 = np.array([d[test_set_name][key] for d in dir_2_data[dir_1]["per_simulated_participant_metrics"]])
            scores_2 = np.array([d[test_set_name][key] for d in dir_2_data[dir_2]["per_simulated_participant_metrics"]])

            if "pvq" in test_set_name:
                scores_1 = per_part_normalized_scores(dir_2_data, dir_1, test_set_name, key)
                scores_2 = per_part_normalized_scores(dir_2_data, dir_2, test_set_name, key)

            correlation = compute_correlation(scores_1, scores_2, spearman, insert_dummy=args.insert_dummy)

            corrs[key].append(correlation)

            key_dir_corrs[key][dir_1].append(correlation)
            key_dir_corrs[key][dir_2].append(correlation)

        mean_ = np.mean([corrs[k][-1] for k in keys])
        if mean_ < 0.53:
            prefix = "\\grayrow "
        else:
            prefix = ""

        # print("\t\t".join([f"{corrs[k][-1]:.2f}" for k in keys] + [f"{mean_:.2f}"]))
        latex_table_row = (prefix + lab_latex).ljust(35, ' ') + "& " + " & ".join([f"{corrs[k][-1]:.2f}" for k in keys] + [f"{mean_:.2f}"]) + " \\\\"
        latex_table_row = latex_table_row.replace("_", "\\_")
        print(latex_table_row)

    key_rank_order_stabilities = {}
    for key in keys:
        key_rank_order_stabilities[key] = np.mean(corrs[key])

    mean_stabilities = list(key_rank_order_stabilities.values())

    mean_rank_order_stability = np.mean(mean_stabilities)
    latex_table_row = "\\bf Mean".ljust(35) + "& " + " & ".join([f"{s:.2f}" for s in mean_stabilities] + [colored(f"{mean_rank_order_stability:.2f}", "blue")]) + " \\\\"
    print(latex_table_row)

    # print("\nSimulated participant order stability due to perspective change (avg over context changes)")
    # print_dict_values(permutation_order_stabilities)

    print(f"\nAverage over context changes ({len(directories)}) and values ({len(keys)})")
    all_corrs = list(itertools.chain(*corrs.values()))
    print_aggregated_correlation_stats(all_corrs)
    key_dir_corrs = {
        k: {
          d: np.mean(v_) for d, v_ in v.items()
        } for k, v in key_dir_corrs.items()}

    return mean_rank_order_stability, key_rank_order_stabilities, key_dir_corrs


def compute_correspondence(dir_2_data):
    # normalized_evaluation_data = []
    notnorm_evaluation_data = []

    primary_value_alignments = []
    mean_vars = []
    labels = []

    for dir_i, (dir, data) in enumerate(dir_2_data.items()):

        labels.append(dir_to_label(dir))
        # normalized_evaluation_data.append([])
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

        normalizing_constants = {
            "pvq_male": defaultdict(lambda: 5),
            "pvq_female": defaultdict(lambda: 5),
            "pvq_auto": defaultdict(lambda: 5),
            "hofstede": {
                "Power Distance": 2 * 300,
                "Individualism": 2 * 350,
                "Masculinity": 2 * 350,
                "Uncertainty Avoidance": 2 * 325,
                "Long-Term Orientation": 2 * 325,
                "Indulgence": 2 * 375,
            },
            "big5_50": defaultdict(lambda: 50),
            "tolkien_donation": defaultdict(lambda: 1),
            "tolkien_bag": defaultdict(lambda: 1),
        }

        normalizing_offset = {
            "pvq_male": defaultdict(lambda: -1),
            "pvq_female": defaultdict(lambda: -1),
            "pvq_auto": defaultdict(lambda: -1),
            "hofstede": {
                "Power Distance": 300,
                "Individualism": 350,
                "Masculinity": 350,
                "Uncertainty Avoidance": 325,
                "Long-Term Orientation": 325,
                "Indulgence": 375,
            },
            "big5_50": defaultdict(lambda: 0),
            "tolkien_donation": defaultdict(lambda: 0),
            "tolkien_bag": defaultdict(lambda: 0),
        }

        per_value_norm_scores = defaultdict(list)

        for perm_i, perm_metrics in enumerate(data["per_permutation_metrics"]):

            # we normalize by (x+normalizing_offset)/normalizing_constant
            assert len(perm_metrics.keys()) == 1
            test_set_name = list(perm_metrics.keys())[0]
            notnorm_perm_metrics = {val: perm_metrics[test_set_name][val] for val in test_set_values}

            norm_perm_metrics = {
                val: (
                             perm_metrics[test_set_name][val] + normalizing_offset[test_set_name][val]
                     ) / normalizing_constants[test_set_name][val] for val in test_set_values
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

            # normalized_evaluation_data[dir_i].append([norm_perm_metrics[v] for v in test_set_values])
            notnorm_evaluation_data[dir_i].append([notnorm_perm_metrics[v] for v in test_set_values])

        # mean (over values/traits) of vars (over permutations)
        mean_vars.append(np.mean([np.var(norm_scores) for norm_scores in per_value_norm_scores.values()]))

        if primary_values:
            perspective_value_alignment = np.mean(
                primary_value_alignments[-len(data["per_permutation_metrics"]):])

            # dump alignments to json

            dump_json_path = dir + "/alignments.json"
            with open(dump_json_path, 'w') as f:
                json.dump(primary_value_alignments[-len(data["per_permutation_metrics"]):], f)

            print(f"Primary value alignment for {primary_values}: {perspective_value_alignment}.")

    if primary_values:
        # todo: confirm this
        mean_primary_value_alignment = np.mean(primary_value_alignments)
        print(colored(f"Mean primary value alignment (over all): {round(mean_primary_value_alignment, 3)}", "green"))

    # mean over contexts
    # mean_var = np.mean(mean_vars)

    # all_evaluation_data = (n_persp, n_perm, n_values/traits)
    # normalized_evaluation_data = np.array(normalized_evaluation_data)
    notnorm_evaluation_data = np.array(notnorm_evaluation_data)

    # Mean and var of answers
    # per_value_persp_mean = notnorm_evaluation_data.mean(axis=0).mean(axis=0)
    # # avg over permutation
    # per_value_persp_std = notnorm_evaluation_data.std(axis=0).mean(axis=0)
    # # std over both contexts and permutation
    # per_value_std_std = notnorm_evaluation_data.std(axis=(0, 1))
    #
    # print("Per value std of contexts")
    # for v_, m_, std_, std_std_ in zip(test_set_values, per_value_persp_mean, per_value_persp_std,
    #                                   per_value_std_std):
    #     print("{:<15} \tM: {:.2f} \tSD (avg perm): {:.2f} \tSD (sd perm) {:.2f}".format(v_, m_, std_, std_std_))
    # print("{:<15} \tM: {:.2f} \tSD (avg perm): {:.2f} \tSD (sd perm) {:.2f}".format(
    #     "Mean",
    #     np.mean(per_value_persp_mean),
    #     np.mean(per_value_persp_std),
    #     np.mean(per_value_std_std)
    # ))

    # print(f"Mean (over values) Variance (over contexts): {mean_variance}")
    # perm_var = normalized_evaluation_data.var(1).mean()
    # # var(permut) -> mean (persp, values)
    # assert np.isclose(perm_var, mean_var)
    # print(
    #     f"Simulated participant Var - mean (over values/traits x contexts) of var (over part) (*10^3): {round(perm_var * (10 ** 3), 2)}")
    #
    # persp_var = normalized_evaluation_data.mean(1).var(0).mean()
    # # mean(permut) -> var (persp) -> mean (values)
    # print(
    #     f"Context Var - mean (over values/traits) of var (over contexts) of mean (over part) (*10^3): {round(persp_var * (10 ** 3), 2)}")

    return notnorm_evaluation_data, labels


def plot_values(part_scores, keys, ips_part_stabilities=None, ips_part_dir_stabilities=None):

    names_to_plot = [
        # tolkien
        "Gandalf", "Peregrin Took", "Frodo Baggins", "Galadriel", "Legolas", "Gimli", "Aragorn", "Gollum",
        "Gothmog (Balrog)", "Lungorthin", "Durin's Bane", "Thuringwethil", "Shelob", "Morgoth", "Sauron"

        # famous
        "Joseph Stalin", "Martin Luther King", "Marilyn Monroe", "Elvis Presley", "Dalai Lama", "Henry Ford",
        "Nelson Mandela", "Mahatma Gandhi", "Thomas Edison",
    ]

    if set(names_to_plot).intersection(set(part_scores.keys())):
        part_scores = {k: part_scores[k] for k in names_to_plot if k in part_scores}
    else:
        part_scores = dict(random.sample(part_scores.items(), 10))

    n_part_to_plot = len(part_scores)

    # plot data
    n_dirs = len(directories)
    fig, axs = plt.subplots(n_part_to_plot, n_dirs, figsize=(5 * n_dirs, 5 * len(keys)))
    axs = axs.flatten()


    # most stable dir/key on average (a good proxy to show overall correlations of everything)
    # ref_key = max(key_rank_order_stabilities, key=key_rank_order_stabilities.get)
    # dir_stabilities = {dir: np.mean(key_dir_stabilities[k][dir]) for k in keys for dir in directories}
    # ref_dir = max(dir_stabilities, key=dir_stabilities.get)

    # the most stable dir/key pair
    # ref_key = max(key_dir_stabilities, key=lambda d: max(key_dir_stabilities[d].values())) # key with the most stable key-dir pair
    # ref_dir = max(key_dir_stabilities[ref_key], key=key_dir_stabilities[ref_key].get)

    # manual set
    # ref_dir = directories[0]
    # ref_key = keys[0]


    for part_i, part in enumerate(part_scores):

        # # # one ref for each key (row)
        ref_part = part
        # ref_dir = directories[0]
        ref_dir = max(ips_part_dir_stabilities[part], key=ips_part_dir_stabilities[part].get)
        ref_vals = part_scores[ref_part][ref_dir]
        ref_vals_order = sorted(ref_vals, key=ref_vals.get)

        for dir_i, dir in enumerate(directories):

            # plot
            ys = [part_scores[part][dir][v] for v in ref_vals_order]
            xs = range(len(ys))

            ticks, labels = xs, ref_vals_order

            # axs[key_i][dir_i].scatter(xs, ys, s=1)
            axs[part_i*n_dirs + dir_i].plot(xs, ys, linewidth=1)
            axs[part_i*n_dirs + dir_i].set_xticks(ticks, labels, rotation=90, fontsize=5)

            if part == ref_part and dir == ref_dir:

                # bold border
                # for spine in axs[key_i*n_dirs + dir_i].spines.values(): spine.set_linewidth(1.5)

                # background
                axs[part_i*n_dirs + dir_i].set_facecolor('lightgray')

            if dir_i == 0:
                # first collumn
                axs[part_i*n_dirs + dir_i].set_ylabel(f"{part}\n(r={ips_part_stabilities[part]:.2f})", rotation='horizontal', ha="right")

            if part_i == 0:
                # first row
                lab = dir_to_label(dir)
                axs[part_i*n_dirs + dir_i].set_title(lab)

    fig.suptitle(f"{dir_2_data[dir]['args']['experiment_name']} - {dir_2_data[dir]['args']['engine']} ", fontsize=16)
    # plt.subplots_adjust(left=0.08, right=0.99, top=0.90, bottom=0.01, hspace=0.05, wspace=0.1)
    plt.subplots_adjust(left=0.08, right=0.99, top=0.95, bottom=0.05, hspace=0.8, wspace=0.1)

    return mean_rank_order_stability

def plot_population(dir_2_data, keys, key_rank_order_stabilities=None, key_dir_stabilities=None):


    # extract data
    directories = list(dir_2_data.keys())
    plot_data = dict()
    for key in keys:
        plot_data[key] = {}

        for dir in directories:

            scores = np.array([d[test_set_name][key] for d in dir_2_data[dir]["per_simulated_participant_metrics"]])

            if "pvq" in test_set_name:
                # extract per participant average answer
                average_part_answer = np.array([
                    np.array(d[test_set_name])[:, 1].astype(float).mean() for d in dir_2_data[dir]["answers"]
                ])

                scores -= average_part_answer

            plot_data[key][dir] = scores

    # plot data
    n_dirs = len(directories)
    fig, axs = plt.subplots(len(keys), n_dirs, figsize=(5 * n_dirs, 5 * len(keys)))
    axs = axs.flatten()


    # # take the most stable one as ref
    dir_stabilities = None


    # most stable dir/key on average (a good proxy to show overall correlations of everything)
    ref_key = max(key_rank_order_stabilities, key=key_rank_order_stabilities.get)
    dir_stabilities = {dir: np.mean(key_dir_stabilities[k][dir]) for k in keys for dir in directories}
    ref_dir = max(dir_stabilities, key=dir_stabilities.get)

    # the most stable dir/key pair
    # ref_key = max(key_dir_stabilities, key=lambda d: max(key_dir_stabilities[d].values())) # key with the most stable key-dir pair
    # ref_dir = max(key_dir_stabilities[ref_key], key=key_dir_stabilities[ref_key].get)

    # manual set
    # ref_dir = directories[0]
    # ref_key = keys[0]

    names_to_plot = [
        # tolkien
        "Gandalf", "Peregrin Took", "Frodo Baggins", "Galadriel", "Legolas", "Gimli", "Aragorn", "Gollum",
        "Gothmog (Balrog)", "Lungorthin", "Durin's Bane", "Thuringwethil", "Shelob", "Morgoth", "Sauron"

        # famous
                                                                                                "Joseph Stalin", "Martin Luther King", "Marilyn Monroe", "Elvis Presley", "Dalai Lama", "Henry Ford",
        "Nelson Mandela", "Mahatma Gandhi", "Thomas Edison",
    ]
    # names_to_plot = [
    #     # tolkien
    #     "Gandalf", "Sauron", "Peregrin Took", "Frodo Baggins", "Morgoth", "Galadriel", "Legolas", "Gimli", "Aragorn", "Shelob", "Gollum",
    #     # famous
    #     "Joseph Stalin", "Martin Luther King", "Marilyn Monroe", "Elvis Presley", "Dalai Lama", "Henry Ford",
    #     "Nelson Mandela", "Mahatma Gandhi", "Thomas Edison",
    # ]

    for key_i, key in enumerate(keys):

        # # one ref for each key (row)
        ref_key = key
        ref_dir = max(key_dir_stabilities[ref_key], key=key_dir_stabilities[ref_key].get)

        ref_pop = plot_data[ref_key][ref_dir]
        ref_indices = ref_pop.argsort()

        for dir_i, dir in enumerate(directories):
            # reorder
            lab = dir_to_label(dir)
            # plot
            ys = plot_data[key][dir][ref_indices]
            xs = range(len(ys))

            reordered_names = list(np.array(dir_2_data[dir]['simulated_population'])[ref_indices])

            # famous people have ( year of birth - year of death), here so filter it
            def extract_name_brackets(text):
                if text is None:
                    return ""
                match = re.match(r'(.*?) \(.*\)', text)
                return match.group(1) if match else text

            reordered_names = [extract_name_brackets(n) for n in reordered_names]

            # filter only salient names
            if set(names_to_plot).intersection(set(reordered_names)):
                ticks, pers_labels = zip(*[(t, l) for t, l in enumerate(reordered_names) if l in names_to_plot])
            else:
                ticks, pers_labels = [], []


            # axs[key_i][dir_i].scatter(xs, ys, s=1)
            axs[key_i*n_dirs + dir_i].plot(xs, ys, linewidth=1)
            axs[key_i*n_dirs + dir_i].set_xticks(ticks, pers_labels, rotation=90, fontsize=8)

            bad_guys = ["Gollum", "Sauron", "Saruman", "Smaug", "Morgoth", "Shelob", "Gríma Wormtongue", "Ungoliant", "Thuringwethil", "Durin's Bane", "Lungorthin"]
            good_guys = ["Gandalf", "Aragorn", "Celeborn", "Galadriel", "Tom Bombadil", "Elrond", "Frodo Baggins", "Finrod Felagund", "Glorfindel", "Goldberry", "Bilbo Baggins", "Faramir", "Éowyn", "Samwise Gamgee", "Fëanor", "Théoden", "Boromir", "Túrin Turambar", "Thranduil", "Beorn", "Arwen", "Halbarad", "Fingon", "Fingolfin", "Celebrimbor", "Gil - galad", "Meriadoc Brandybuck", "Treebeard", "Radagast", "Elendil", "Éomer", "Legolas", "Húrin", "Thorin Oakenshield", "Peregrin Took", "Thingol", "Eärendil", "Elwing", "Lúthien", "Beren", "Tuor", "Idril", "Finwë", "Míriel", "Melian", "Balin", "Gimli"]

            for lab_i, p_lab in enumerate(pers_labels):
                if p_lab in bad_guys:
                    axs[key_i*n_dirs+dir_i].get_xticklabels()[lab_i].set_color("red")
                elif p_lab in good_guys:
                    axs[key_i*n_dirs+dir_i].get_xticklabels()[lab_i].set_color("green")


            for t in ticks:
                axs[key_i*n_dirs + dir_i].axvline(t, linestyle=":", linewidth=0.5)
            axs[key_i*n_dirs + dir_i].scatter(ticks, [ys[t] for t in ticks], s=7, c="r")


            if key == ref_key and dir == ref_dir:

                # bold border
                # for spine in axs[key_i*n_dirs + dir_i].spines.values(): spine.set_linewidth(1.5)

                # background
                axs[key_i*n_dirs + dir_i].set_facecolor('lightgray')

            if dir_i == 0:
                # first collumn
                axs[key_i*n_dirs + dir_i].set_ylabel(key + f"\n(r={key_rank_order_stabilities[key]:.2f})" if key_rank_order_stabilities else "", rotation='horizontal', ha="right")

            if key_i == 0:
                # first row
                # axs[key_i*n_dirs + dir_i].set_title(lab + f" (r={dir_stabilities[dir]:.2f})" if dir_stabilities else "")
                axs[key_i*n_dirs + dir_i].set_title(lab)

    fig.suptitle(f"{dir_2_data[dir]['args']['experiment_name']} - {dir_2_data[dir]['args']['engine']} ", fontsize=16)
    # plt.subplots_adjust(left=0.08, right=0.99, top=0.90, bottom=0.01, hspace=0.05, wspace=0.1)
    plt.subplots_adjust(left=0.08, right=0.99, top=0.90, bottom=0.1, hspace=0.65, wspace=0.1)

    return mean_rank_order_stability

def statistical_analysis_and_cohen_ds(notnorm_evaluation_data, dir_2_data):
    p_limit = 0.05
    # p_limit = 0.005
    # p_limit = 0.001

    # group_p_limit = 0.001
    group_p_limit = p_limit / len(test_set_values)  # additional bonferroni correction

    print("testing p-value: ", group_p_limit)
    if "pvq" in test_set_name:
        # PVQ centering - is this reccomended in this manual? # todo: do properly all questions

        n_values = notnorm_evaluation_data.shape[2]
        notnorm_evaluation_data = notnorm_evaluation_data - np.repeat(notnorm_evaluation_data.mean(2)[:,:, np.newaxis], n_values, axis=2)

    cohens_ds = defaultdict(list)

    for val_i, val in enumerate(test_set_values):
        # value_data = normalized_evaluation_data[:, :, val_i] # value_data = (n_perspectives, n_permutations)
        value_data = notnorm_evaluation_data[:, :, val_i] # value_data = (n_contexts, n_permutations)

        print(f"\n----------------{val}---------------------")

        # standard ANOVA
        if value_data.var() == 0:
            pvalue = np.inf
        else:
            _, pvalue = stats.f_oneway(*value_data)

        if pvalue >= group_p_limit:
            different_distr.append(False)
            print(colored(f"p(ANOVA): {pvalue} > {group_p_limit}", "red"))
            posthoc_test = False
        else:
            different_distr.append(True)
            print(colored(f"p(ANOVA): {pvalue} < {group_p_limit}", "green"))
            posthoc_test = True

        # do posthocs and compute cohen's ds
        pairs_ind = list(itertools.combinations(range(len(value_data)), 2))

        if posthoc_test:
            tukey_res = tukey_hsd(*value_data)  # Tukey test

        latex_labels = []
        print("\tTukey (p limit) :", group_p_limit)
        for pair_ind in pairs_ind:

            # cohen's d
            d = cohen_d(value_data[pair_ind[0]], value_data[pair_ind[1]])
            cohens_ds[val].append(d)

            latex_labels.append(f"{labels[pair_ind[0]]} - {labels[pair_ind[1]]} & ")

            # posthoc
            if posthoc_test:
                res_pvalue = tukey_res.pvalue[pair_ind[0], pair_ind[1]]
                if res_pvalue < p_limit:
                    print("\t{} - {} : p -> {:.8f} d -> {:.8f}".format(
                        labels[pair_ind[0]],
                        labels[pair_ind[1]],
                        res_pvalue, d)
                    )


    # Mean-level
    print(colored("\n\n--------------------------------------------------", "green"))
    print(colored("MEAN-LEVEL STABILITY ", "green"))
    print(colored("--------------------------------------------------", "green"))

    print(" & ".join(cohens_ds.keys()) + " Mean \\\\")
    for lab_i, latex_lab in enumerate(latex_labels):
        value_stabilities = [v[lab_i] for v in cohens_ds.values()]
        value_stabilities = value_stabilities + [np.mean(np.abs(value_stabilities))]
        latex_row = latex_lab + " & ".join([f"{v:.2f}" for v in value_stabilities]) + " \\\\"
        print(latex_row.replace("_", "\\_"))

    avg_abs_cohens_ds = {k: np.mean(np.abs(v)) for k, v in cohens_ds.items()}

    mean_mean_level_stability = np.mean(list(avg_abs_cohens_ds.values()))
    latex_str = "\\bf Mean & " + \
                " & ".join([f"{v:.2f}" for v in avg_abs_cohens_ds.values()]) + \
                " & " + colored(f"{mean_mean_level_stability:.2f}", "blue") + " \\\\"

    print(latex_str)

    print("Average absolute cohen's ds:")
    print_dict_values(avg_abs_cohens_ds)


    # plot bars
    if args.plot_mean:

        for distr_i, different in enumerate(different_distr):
            if not different:
                # anova didn't reject - same distr
                plt.bar(distr_i, 300, width=all_bars_width*1.3, color="lightgray")
                plt.bar(distr_i, -300, width=all_bars_width*1.3, color="lightgray")

        for i, directory in enumerate(directories):

            if not os.path.isdir(directory):
                continue

            results_json_path = os.path.join(directory, 'results.json')

            if not os.path.isfile(results_json_path):
                continue

            data = dir_2_data[directory]

            offset = -all_bars_width/2 + (i/num_dirs)*all_bars_width
            keys, sorted_keys, keys_vals = plot_baseline(data, ax, directory, offset, bar_width=bar_width, min_bar_size=0.05)
            assert keys == test_set_values

    print()
    return test_set_values, mean_mean_level_stability

def is_strictly_increasing(lst):
    return all(x <= y for x, y in zip(lst, lst[1:]))


def print_dict_values(d):
    print("\t".join([f"{k:<10}" for k in list(d.keys()) + ["Mean"]]))
    print("\t\t".join([f"{np.round(s, 2):.2}" for s in list(d.values()) + [np.mean(list(d.values()))]]))


warnings.filterwarnings('error', category=ConstantInputWarning)
def compute_correlation(scores_1, scores_2, spearman, insert_dummy):

    # trick to overcome bad collapse evaluation
    # assert not insert_dummy # don't use this
    if insert_dummy:
        scores_1 = np.insert(scores_1, 0, [-100])
        scores_2 = np.insert(scores_2, 0, [-100])

    try:
        if spearman:
            correlation, _ = spearmanr(scores_1, scores_2)
        else:
            correlation, _ = pearsonr(scores_1, scores_2)

    except ConstantInputWarning:
        # return np.nan

        # not ideal but no other way (apart from dummy insert above)
        if len(set(scores_1)) == 1 and len(set(scores_2)) == 1:
            # both constant
            # correlation = 1.0
            print(f"Collapse setting 1.")
            return np.nan
        else:
            # one constant  # todo: return to this is 0, above is nan
            correlation = 0.0
            # print(f"Collapse setting 0.")
            # return np.nan

    return correlation


def dir_to_label(directory):

    if "format_chat_simulate_conv" in directory:
        label = extract_value(directory, "_simulate_conv_")

    elif "simulate_conv" in directory:
        label = extract_value(directory, "_simulate_conv_")

    elif "weather" in directory:
        label = extract_value(directory, "_weather_")

    elif "no_profile" in directory:
        label = extract_value(directory, "_format_")

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


def print_correlation_stats(cs, header=False, label="\t", color=None):
    from scipy.stats import kurtosis, skew

    statistics = {
        "Mean": np.mean(cs),
        "Median": np.median(cs),
        "STD": np.std(cs),
        "Min": np.min(cs),
        "Max": np.max(cs),
    }

    if header:
        headers = "\t".join(statistics.keys())
        print(f"\t\t{headers}")

    values = f"{label}".ljust(20, ' ') + "\t".join(f"{val:.2f}" for val in statistics.values())

    if color:
        print(colored(f"{values}", color))
    else:
        print(f"{values}")

    return statistics


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


def plot_baseline(data, ax, directory, offset, subj=None, bar_width=1.0, min_bar_size=0.1):

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

    keys = list(draw_metrics.keys())

    key_indices = {key: i for i, key in enumerate(keys)}

    perm_vals = []
    for perm_m in data['per_permutation_metrics']:
        perm_vals.append([perm_m[test_set_name][key] for key in keys])

    perm_vals = np.array(perm_vals)

    if "pvq" in test_set_name:
        # from IPython import embed; embed();
        # centering PVQ
        # todo: fix because this mean is not perfect (should be mean of all answers)
        # this is mean of scores - universalism has 5 questions the rest 4
        perm_vals = perm_vals - np.repeat(perm_vals.mean(1)[:, np.newaxis], 10, axis=1)

        # np.array([[ans[1] for ans in part['pvq_male']] for part in data['answers']])

    values = perm_vals.mean(0)
    # errs = perm_vals.std(0)

    errs = sem(perm_vals)

    label = dir_to_label(directory)

    keys_vals = dict(zip(keys, values))
    sorted_keys = [k for k, v in sorted(keys_vals.items(), key=lambda item: item[1], reverse=True)]

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

    assert len(errs) == len(values)
    ax.bar(x_values, values, yerr=errs, label=label, width=bar_width, color=None)

    # set y-axis limits
    if "pvq" in test_set_name:
        # ax.set_ylim([-3, 3]) # append
        ax.set_ylim([-2.9, 2.9])

    elif test_set_name == "hofstede":
        # ax.set_ylim([-350, 350]) # append
        ax.set_ylim([-130, 230])

    elif test_set_name == "big5_50":
        ax.set_ylim([0, 55])

    elif test_set_name == "big5_100":
        ax.set_ylim([0, 110])

    elif test_set_name == "tolkien_ultimatum":
        ax.set_ylim([0, 1])

    elif test_set_name == "tolkien_donation":
        ax.set_ylim([0, 10])

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


if __name__ == '__main__':
    import argparse

    # normalized_evaluation_data = []
    # notnorm_evaluation_data = []
    vals = []

    parser = argparse.ArgumentParser()
    parser.add_argument('directories', nargs='+', help='directories containing results.json files')
    parser.add_argument('--no-ips', action="store_true")
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
    parser.add_argument('--neutral-ranks', action="store_true")
    parser.add_argument('--neutral-dir', type=str)

    # "results/sim_conv_pvq_permutations_msgs/Mixtral-8x7B-Instruct-v0.1/9_msgs/_seed/results_sim_conv_permutations_Mixtral-8x7B-Instruct-v0.1/pvq_test_Mixtral-8x7B-Instruct-v0.1_data_pvq_pvq_auto__permutations_50_permute_options_5_no_profile_True_format_chat___2024_02_14_20_47_27"
    parser.add_argument('--default-profile', type=str, default=None)
    parser.add_argument('--paired-dirs', nargs='+', default=None)

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
        "format_chat___",
    ]

    args.directories = [d for d in args.directories if not any([i in d for i in ignore])]
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

    # compute correspondence
    notnorm_evaluation_data, labels = compute_correspondence(dir_2_data)

    keys, mean_mean_level_stability = statistical_analysis_and_cohen_ds(notnorm_evaluation_data=notnorm_evaluation_data, dir_2_data=dir_2_data)

    assert directories == list(dir_2_data.keys())

    mean_rank_order_stability, key_rank_order_stabilities, key_dir_stabilities = compute_rank_order_stability(dir_2_data=dir_2_data, keys=keys)

    if len(keys) >= 2 and not args.no_ips:
        mean_ipsative_stability, part_scores, ips_part_stabilities, ips_part_dir_stabilities, all_ips_corrs = compute_ipsative_stability(dir_2_data=dir_2_data, keys=keys)
    else:
        all_ips_corrs=np.nan
        mean_ipsative_stability = np.nan
        all_corrs = np.nan
        print(f"IPsative stability is not computed because there only one metric {keys}.")

    if args.default_profile:
        dir_2_data_neut_prof = load_data([args.default_profile])
        assert len(dir_2_data_neut_prof.keys()) == 1
        dir_neut_prof = list(dir_2_data_neut_prof.keys())[0]

        # per participant normalize the neutral profile
        normalized_scores = []
        for key in keys:

            scores_neut_prof = np.array([d[test_set_name][key] for d in dir_2_data_neut_prof[dir_neut_prof]["per_simulated_participant_metrics"]])

            if "pvq" in test_set_name:
                # todo: remove
                # extract per participant average answer
                average_part_answer_neut_prof = np.array([
                    np.array(d[test_set_name])[:, 1].astype(float).mean() for d in dir_2_data_neut_prof[dir_neut_prof]["answers"]
                ])
                scores_neut_prof_ = scores_neut_prof - average_part_answer_neut_prof
                scores_neut_prof = per_part_normalized_scores(dir_2_data_neut_prof, dir_neut_prof, test_set_name, key)
                assert all(scores_neut_prof_ == scores_neut_prof)

                normalized_scores.append(scores_neut_prof.copy())

        normalized_scores = np.array(normalized_scores)
        assert normalized_scores.shape == (len(keys), len(scores_neut_prof))
        ranks = rankdata(normalized_scores, axis=0)
        # default profile is the mean rank
        default_profile = ranks.mean(axis=1)


        # compute the ipsative correlation with the default profile
        (
            mean_ipsative_stability_default_profile,
            part_scores_default_profile,
            ips_part_stabilities_default_profile,
            ips_part_dir_stabilities_default_profile,
            all_ips_corrs_default_profile
        ) = compute_ipsative_stability(
            dir_2_data=dir_2_data,
            keys=keys,
            default_profile=default_profile
        )
        print("Ipsative stability with default profile: ", mean_ipsative_stability_default_profile)

    else:
        mean_ipsative_stability_default_profile = None
        all_ips_corrs_default_profile = None


    neutral_rank_order_stability = None
    if args.neutral_ranks:
        print()
        print(colored("------------------------", "green"))
        print(colored("Neutral Rank-Order stability", "green"))
        print(colored("------------------------", "green"))

        # find the neutral context dir

        neutral_dir = glob.glob(args.neutral_dir + "/*/*chat___*")[0]
        dir_2_data_neutral = load_data([neutral_dir])

        # assert no topic
        assert dir_2_data_neutral[neutral_dir]['args']['simulate_conversation_theme'] is None

        values_names = [
            "Benevolence", "Universalism", "Power", "Achievement", "Tradition",
            "Conformity", "Security", "Self-Direction", "Stimulation", "Hedonism"
        ]

        mean_stability = []
        for value in values_names:

            stab_, _, _ = compute_paired_rank_order_stability(
                dir_2_data, dir_2_data_neutral,
                value, value,
                "pvq_auto", "pvq_auto",
                directories_1=dir_2_data.keys(), directories_2=[neutral_dir]*len(dir_2_data)
            )

            mean_stability.append(stab_)

        neutral_rank_order_stability = np.mean(mean_stability)

        print(f"Mean stability w neutral rank: {neutral_rank_order_stability}")

    proxy_stability = None
    if args.paired_dirs:

        print("\n\nPaired Rank-Order stability\n")

        for dir_, ben_ in zip(args.directories, args.paired_dirs):
            # model and seed should be analogous
            assert dir_.split("/")[2:4] == ben_.split("/")[2:4]

            # get conversation theme
            assert dir_.split("chat__")[1].split("202")[0] == ben_.split("chat__")[1].split("202")[0]

        dir_2_data_neut_prof = load_data(args.paired_dirs)

        values_names = ["Benevolence", "Universalism", "Power", "Achievement", "Tradition", "Conformity", "Security",
                        "Self-Direction", "Stimulation", "Hedonism"]

        proxy_stability = {}
        for value in values_names:
            mean_stability = []
            for race in ["elves", 'dwarves', 'orcs', 'humans', 'hobbits']:
                mean_paired_rank_order_stability, _, _ = compute_paired_rank_order_stability(
                    dir_2_data, dir_2_data_neut_prof,
                    value, f"Donation {race}",
                    "pvq_auto", "tolkien_donation"
                )
                # print(f"{race}: {mean_paired_rank_order_stability}")
                mean_stability.append(mean_paired_rank_order_stability)

            proxy_stability[value] = np.mean(mean_stability)

            print(f"Mean {value} stability: {proxy_stability[value]}.")


    print()
    print(colored("------------------------", "green"))
    print(colored("Aggregated metrics", "green"))
    print(colored("------------------------", "green"))

    print("Mean-Level\tRank-Order\tIpsative")
    print(f"{mean_mean_level_stability:.4f}\t\t{mean_rank_order_stability:.4f}\t\t{mean_ipsative_stability:.4f}")

    if args.result_json_stdout:
        sys.stdout = sys.__stdout__
        outputs = {
            "Mean-Level": mean_mean_level_stability,
            "Rank-Order": mean_rank_order_stability,
            "Ipsative": mean_ipsative_stability,
            "All_Ipsative_corrs": all_ips_corrs,
            "Ipsative_default_profile": mean_ipsative_stability_default_profile,
            "All_Ipsative_corrs_default_profile": all_ips_corrs_default_profile,
            "Proxy_stability": proxy_stability,
            "Neutral_Rank-Order": neutral_rank_order_stability,
        }

        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return json.JSONEncoder.default(self, obj)

        print(json.dumps(outputs, cls=NumpyEncoder))

    if args.plot_ranks:
        plot_population(dir_2_data, keys, key_rank_order_stabilities=key_rank_order_stabilities, key_dir_stabilities=key_dir_stabilities)

        if not args.plot_dont_show:
            plt.show()

        if args.plot_save:
            # for ext in ["png", "svg"]:
            for ext in ["png"]:
                savepath = f"visualizations/{args.filename}.{ext}"
                print(f"Saved to: {savepath}")
                # plt.tight_layout()
                plt.savefig(savepath)

    if args.plot_ips:
        plot_values(part_scores, keys, ips_part_stabilities=ips_part_stabilities, ips_part_dir_stabilities=ips_part_dir_stabilities)

        if not args.plot_dont_show:
            plt.show()

        if args.plot_save:
            # for ext in ["png", "svg"]:
            for ext in ["png"]:
                savepath = f"visualizations/{args.filename}.{ext}"
                print(f"Saved to: {savepath}")
                # plt.tight_layout()
                plt.savefig(savepath)

    if args.plot_mean:

        # PLOT
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels(keys, rotation=90)
        ax.tick_params(axis='both', which='both', labelsize=30)

        if args.plot_save:
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
