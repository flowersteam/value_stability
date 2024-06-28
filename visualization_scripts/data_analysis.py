import copy
import os
import json
import warnings
import random
import glob
from pprint import pprint

import matplotlib.pyplot as plt
import re
import numpy as np
from termcolor import colored
import itertools
from scipy.stats import rankdata, spearmanr, ConstantInputWarning
from collections import defaultdict

from sklearn.preprocessing import StandardScaler  # to standardize the features
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.manifold import MDS

import pandas as pd
import pingouin as pg


label_parser = lambda d: d.split("/")[-1].split("_202")[0]

def conduct_cfa(dir_2_data, test_set_name):

    SRMRs, RMSEAs, CFIs, TLIs = {}, {}, {}, {}

    for dir in dir_2_data.keys():

        data_answers = np.array([np.array(d[test_set_name])[:, 1].astype(float) for d in dir_2_data[dir]["answers"]])

        # Create column names for the items
        columns = [f'item{i + 1}' for i in range(data_answers.shape[1])]

        # Convert the NumPy array to a Pandas DataFrame
        df = pd.DataFrame(data_answers, columns=columns)

        # Save the DataFrame to a CSV file
        df.to_csv('dummy_data.csv', index=False)

        # Path to the data file
        import subprocess
        # Run the R script
        questionnare = "SVS" if "svs" in dir else "PVQ"
        result = subprocess.run(["Rscript", "cfa.R", 'dummy_data.csv', questionnare], capture_output=True, text=True)

        # Check if the script ran successfully
        if result.returncode != 0:
            CFIs[dir] = 0
            TLIs[dir] = 0
            SRMRs[dir] = 1
            RMSEAs[dir] = 1
            print(result.stderr)
        else:

            try:
                # Parse the JSON output from R
                fit_indices = json.loads(result.stdout)
                print(f"D: {dir}")
                print(fit_indices)

                CFIs[dir] = fit_indices["cfi"][0]
                TLIs[dir] = fit_indices["tli"][0]
                SRMRs[dir] = fit_indices["srmr"][0]
                RMSEAs[dir] = fit_indices["rmsea"][0]
            except:
                CFIs[dir] = 0
                TLIs[dir] = 0
                SRMRs[dir] = 1
                RMSEAs[dir] = 1

    return SRMRs, RMSEAs, CFIs, TLIs


def average_correlation_table(correlation_table):
    keys = list(correlation_table.keys())
    dirs = list(correlation_table[keys[0]])
    avg_correlation_table = {}

    for d in dirs:
        avg_correlation_table[d] = defaultdict(dict)
        for d_ in dirs:
            avg_correlation_table[d][d_] = np.mean([correlation_table[k][d][d_] for k in keys])

    return avg_correlation_table


def plot_pairwise_correlations(correlations_matrix, label_parser=None, title=None, savepath=None):
    labels = sorted(correlations_matrix.keys())

    parsed_label_order = [
        'theme_chess',
        'theme_joke',
        'theme_year',
        'theme_santa',
        'theme_history',
        'theme_bicycle',
        'theme_poem',
        'theme_chord',
        'theme_grammar',
        'theme_cooking',
        'theme_weather',
        'theme_code',
        'theme_britney',
        'theme_bjj'
    ]

    parsed_label_order = [
        "chunk_mix_0",
        "chunk_0",
        "chunk_1",
        "chunk_2",
        "chunk_3",
        "chunk_4",
        "chunk_no_conv",
        "chunk_chess_0",
        "chunk_grammar_1",
        'chunk_grammar_19_msgs_1',
    ]

    try:
        # order labels based on parsed label order
        labels = sorted(
            labels,
            key=lambda l: parsed_label_order.index(label_parser(l))
        )
        labels = [l.removeprefix("theme_") for l in labels]

    except:
        pass

    matrix = np.zeros((len(labels), len(labels)), dtype=float)

    for key_y, predictions in correlations_matrix.items():
        key_y_ind = labels.index(key_y)
        for key_x, cor in predictions.items():
            key_x_ind = labels.index(key_x)
            matrix[key_y_ind][key_x_ind] = cor

    fig, ax = plt.subplots()
    fig.set_size_inches(10, 10)
    im = ax.imshow(matrix, cmap="viridis", vmin=0, vmax=1)

    # Set ticks
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))

    if label_parser is not None:
        parsed_labels = list(map(label_parser, labels))
    else:
        parsed_labels = labels

    ax.set_xticklabels(parsed_labels)
    ax.set_yticklabels(parsed_labels)

    # Rotate the tick labels and set their alignment
    # fig.colorbar(im, ax=ax, shrink=0.5)

    plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")
    # plt.setp(ax.get_xticklabels())

    # Loop over data dimensions and create text annotations
    for i in range(len(labels)):
        for j in range(len(labels)):
            text = ax.text(j, i, matrix[i, j].round(2), ha="center", va="center", color="black")

    if title:
        ax.set_title(title, fontsize=20)

    fig.tight_layout()
    if savepath is not None:
        plt.savefig(savepath)
        print(f"Saved to {savepath}")
    else:
        plt.show()


def per_part_normalized_scores(dir_2_data, dir, test_set_name, key, return_all_answers=False):
    assert "pvq" in test_set_name or "svs" in test_set_name
    scores = np.array([d[test_set_name][key] for d in dir_2_data[dir]["per_simulated_participant_metrics"]])

    all_answers = np.array([np.array(d[test_set_name])[:, 1].astype(float) for d in dir_2_data[dir]["answers"]])
    average_part_answer = all_answers.mean(axis=1)

    centered_scores = scores - average_part_answer

    if return_all_answers:
        centered_answers = all_answers - average_part_answer[:, None]
        return centered_scores, centered_answers

    else:
        return centered_scores


def extract_test_set_name(dir_2_data):
    test_set_names = set(itertools.chain(*[v['metrics'].keys() for v in dir_2_data.values()]))
    assert len(test_set_names) == 1
    test_set_name = list(test_set_names)[0]
    return test_set_name


def get_model_name(directories):
    models = []
    for d in directories:
        with open(os.path.join(d, "results.json"), 'r') as f:
            models.append(json.load(f)['args']['engine'])
    assert set(models) == {models[0]}
    return models[0]


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
        results_json_path = os.path.join(directory, 'results.json')

        if not os.path.isdir(directory) or not os.path.isfile(results_json_path):
            continue

        with open(results_json_path, 'r') as f:
            dir_data = json.load(f)


        # parse svs to pvq label
        def svs_2_pvq_dict(d):

            new = {}
            for k, v in d.items():
                if isinstance(v, dict):
                    v = svs_2_pvq_dict(v)

                elif isinstance(v, list):
                    v_ = []
                    for e in v:
                        if isinstance(e, dict):
                            e = svs_2_pvq_dict(e)
                        v_.append(e)
                    v = v_

                new[k.replace('svs', 'pvq_auto')] = v
            return new

        dir_data = svs_2_pvq_dict(dir_data)

        data[directory] = dir_data

    return data


def compute_ipsative_stability(dir_2_data, keys, default_profile=None):

    # IPSATIVE CORRELATIONS
    print(colored("\n\n--------------------------------------------------", "green"))
    print(colored(f"IPSATIVE CORELATIONS {'WITH DEFAULT PROFILE' if default_profile is not None else ''}", "green"))
    print(colored("--------------------------------------------------", "green"))

    # order of contexts - avg over permutations last (average r)
    print("Correlation (between contexts) - values order (average r)")
    key_orders = defaultdict(dict)
    part_scores = defaultdict(lambda: defaultdict(dict))

    directories = list(dir_2_data.keys())

    population = dir_2_data[directories[0]]['simulated_population']

    if population[0] is None:
        if set(population) == {None}:
            population = list(range(len(population)))

    for p_i, part in enumerate(population):
        for d in directories:
            for key in keys:
                scores = dir_2_data[d]["per_permutation_metrics"][p_i][test_set_name][key]

                if "pvq" in test_set_name or "svs" in test_set_name:
                    scores = per_part_normalized_scores(dir_2_data, d, test_set_name, key)[p_i]

                part_scores[p_i][d][key] = scores

            # keys in order
            key_orders[p_i][d] = [k for k, v in sorted(part_scores[p_i][d].items(), key=lambda item: item[1])]

    headers = "\t".join(["Mean", "Median", "STD", "Min", "Max"])
    print("".ljust(20, " ") + f"{headers}")

    all_corrs = list()

    dir_stabilities = defaultdict(lambda : defaultdict(list))
    ips_part_stabilities = defaultdict(list)

    if default_profile is not None:
        for dir_ in directories:
            population_correlations = []

            for p_i, part in enumerate(population):
                scores_ = [part_scores[p_i][dir_][k] for k in keys]
                corr = compute_correlation(scores_, default_profile)

                dir_stabilities[p_i][dir_].append(corr)

                ips_part_stabilities[p_i].append(corr)

                population_correlations.append(corr)

            label = f"{dir_to_label(dir_)}:"
            print_correlation_stats(population_correlations, label=label)

            all_corrs.append(population_correlations)

    else:
        for dir1, dir2 in itertools.combinations(directories, 2):
            population_correlations = []

            for p_i, part in enumerate(population):

                scores_1 = [part_scores[p_i][dir1][k] for k in keys]
                scores_2 = [part_scores[p_i][dir2][k] for k in keys]

                corr = compute_correlation(scores_1, scores_2)

                dir_stabilities[p_i][dir1].append(corr)
                dir_stabilities[p_i][dir2].append(corr)

                ips_part_stabilities[p_i].append(corr)

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

def compute_paired_rank_order_stability(dir_2_data_1, dir_2_data_2, key_1, key_2, test_set_name_1, test_set_name_2, verbose=False):

    if verbose:
        print(colored("\n\n--------------------------------------------------", "green"))
        print(colored("PAIRED RANK ORDER STABILITY", "green"))
        print(colored("--------------------------------------------------", "green"))

        print("Rank Order stability - Correlation in simulated participant order (between two contexts)\n")

    corrs = []

    # order paired_dirs according to conv topics in directories
    try:
        directories_1, dirs_1_themes = zip(
            *[(dir_1, data['args']['simulated_conversation_theme']) for dir_1, data in dir_2_data_1.items()]
        )
        theme_to_dir_2 = {data['args']['simulated_conversation_theme']: dir_2 for dir_2, data in dir_2_data_2.items()}
    except:
        directories_1, dirs_1_themes = zip(
            *[(dir_1, data['args']['simulate_conversation_theme']) for dir_1, data in dir_2_data_1.items()]
        )
        theme_to_dir_2 = {data['args']['simulate_conversation_theme']: dir_2 for dir_2, data in dir_2_data_2.items()}

    directories_2 = [theme_to_dir_2[theme] for theme in dirs_1_themes]

    # pair directories
    dir_pairs = zip(directories_1, directories_2)
    dir_2_data = {**dir_2_data_1, **dir_2_data_2}

    key_dir_corrs = defaultdict(lambda: defaultdict(list))

    for dir_1, dir_2 in dir_pairs:
        lab_1, lab_2 = dir_to_label(dir_1), dir_to_label(dir_2)

        scores_1 = np.array([d[test_set_name_1][key_1] for d in dir_2_data[dir_1]["per_simulated_participant_metrics"]])
        scores_2 = np.array([d[test_set_name_2][key_2] for d in dir_2_data[dir_2]["per_simulated_participant_metrics"]])

        if "pvq" in test_set_name_1 or "svs" in test_set_name_1:
            scores_1 = per_part_normalized_scores(dir_2_data, dir_1, test_set_name_1, key_1)

        if "pvq" in test_set_name_2 or "svs" in test_set_name_2:
            scores_2 = per_part_normalized_scores(dir_2_data, dir_2, test_set_name_2, key_2)

        correlation = compute_correlation(scores_1, scores_2)

        if verbose:
            print(f"{lab_1} - {lab_2} : {correlation}")

        corrs.append(correlation)

        key_dir_corrs[key_1][dir_1].append(correlation)
        key_dir_corrs[key_1][dir_2].append(correlation)

    key_rank_order_stabilities = {key_1: np.mean(corrs)}
    mean_stabilities = list(key_rank_order_stabilities.values())

    mean_rank_order_stability = np.mean(mean_stabilities)

    if verbose:
        print(f"\nAverage over context changes ({len(directories)}) and values ({len(keys)})")
        print_aggregated_correlation_stats(corrs)

    key_dir_corrs = {
        k: {
            d: np.mean(v_) for d, v_ in v.items()
        } for k, v in key_dir_corrs.items()}

    return mean_rank_order_stability, key_rank_order_stabilities, key_dir_corrs, corrs


def compute_rank_order_stability(dir_2_data, keys):

    print(colored("\n\n--------------------------------------------------", "green"))
    print(colored("RANK ORDER STABILITY", "green"))
    print(colored("--------------------------------------------------", "green"))

    print("Rank Order stability - Correlation in simulated participant order (between two contexts)\n")
    corrs = defaultdict(list)

    directories = list(dir_2_data.keys())


    key_dir_corrs = defaultdict(lambda: defaultdict(list))

    dir_pairs = itertools.combinations(directories, 2)

    correlation_table = { k:defaultdict(dict) for k in keys}

    for dir_1, dir_2 in dir_pairs:
        lab_1 = dir_to_label(dir_1)
        lab_2 = dir_to_label(dir_2)

        for key in keys:

            scores_1 = np.array([d[test_set_name][key] for d in dir_2_data[dir_1]["per_simulated_participant_metrics"]])
            scores_2 = np.array([d[test_set_name][key] for d in dir_2_data[dir_2]["per_simulated_participant_metrics"]])

            if "pvq" in test_set_name or "svs" in test_set_name:
                scores_1 = per_part_normalized_scores(dir_2_data, dir_1, test_set_name, key)
                scores_2 = per_part_normalized_scores(dir_2_data, dir_2, test_set_name, key)

            correlation = compute_correlation(scores_1, scores_2)

            corrs[key].append(correlation)

            correlation_table[key][dir_1][dir_1]=1.0
            correlation_table[key][dir_2][dir_2]=1.0

            correlation_table[key][dir_1][dir_2] = correlation
            correlation_table[key][dir_2][dir_1] = correlation

            key_dir_corrs[key][dir_1].append(correlation)
            key_dir_corrs[key][dir_2].append(correlation)


    key_rank_order_stabilities = {}
    for key in keys:
        key_rank_order_stabilities[key] = np.mean(corrs[key])

    mean_stabilities = list(key_rank_order_stabilities.values())

    mean_rank_order_stability = np.mean(mean_stabilities)
    print("Per Value Rank-Order stabilities")
    pprint(key_rank_order_stabilities)

    print(f"\nAverage over context changes ({len(directories)}) and values ({len(keys)})")
    all_corrs = list(itertools.chain(*corrs.values()))
    print_aggregated_correlation_stats(all_corrs)
    key_dir_corrs = {
        k: {
          d: np.mean(v_) for d, v_ in v.items()
        } for k, v in key_dir_corrs.items()}

    return mean_rank_order_stability, key_rank_order_stabilities, key_dir_corrs, corrs, correlation_table


def compute_value_structure(dir_2_data, keys):

    print(colored("\n\n--------------------------------------------------", "green"))
    print(colored("VALUE STRUCTURE", "green"))
    print(colored("--------------------------------------------------", "green"))

    # fit SSA
    theoretical_ranks = [
        ('Self-Direction', 1),
        ('Stimulation', 2),
        ('Hedonism', 3),
        ('Achievement', 4),
        ('Power', 5),
        ('Security', 6),
        ('Conformity', 7.5),
        ('Tradition', 7.5),
        ('Benevolence', 9),
        ('Universalism', 10)
    ]

    theoretical_values_in_order = [v[0] for v in theoretical_ranks]
    theoretical_ranks_in_order = [v[1] for v in theoretical_ranks]

    directories = list(dir_2_data.keys())

    key_pairs = list(itertools.combinations(keys, 2))

    intercorrelations = {}
    rs = {}
    angles = {}
    per_dir_structure_correlations = {}
    for dir in directories:

        # compute intercorrelations
        intercorrelations[dir] = defaultdict(dict)

        for key in keys:
            intercorrelations[dir][key][key] = 1.0

        for key_1, key_2 in key_pairs:

            scores_1 = np.array([d[test_set_name][key_1] for d in dir_2_data[dir]["per_simulated_participant_metrics"]])
            scores_2 = np.array([d[test_set_name][key_2] for d in dir_2_data[dir]["per_simulated_participant_metrics"]])

            # not for CFA as it makes them dependant
            # if "pvq" in test_set_name or "svs" in test_set_name:
            #     scores_1 = per_part_normalized_scores(dir_2_data, dir, test_set_name, key_1)
            #     scores_2 = per_part_normalized_scores(dir_2_data, dir, test_set_name, key_2)

            correlation = compute_correlation(scores_1, scores_2)

            intercorrelations[dir][key_1][key_2] = correlation
            intercorrelations[dir][key_2][key_1] = correlation

        if args.print_intercorrelations:
            print(f"D: {dir}")
            print("\t"+"".join([f"{k:.4s}\t" for k in keys]))
            for key in keys:
                print(f"{key:.5s}\t", end="")
                for key_ in keys:
                    print(f"{intercorrelations[dir][key][key_]:.2f}\t", end="")
                print()

        # compute the structure corrleation
        data = np.array([[intercorrelations[dir][v1][v2] for v2 in theoretical_values_in_order] for v1 in theoretical_values_in_order])

        # data = (n_samples, n_features)
        # from IPython import embed; embed();
        # from sklearn.decomposition import FactorAnalysis
        from factor_analyzer import calculate_bartlett_sphericity, calculate_kmo
        from factor_analyzer import Rotator
        # from factor_analyzer import ConfirmatoryFactorAnalyzer
        # model = ConfirmatoryFactorAnalyzer(n_factors=3,
        #                                    method='ml',
        #                                    rotate='varimax')

        # data_pca = PCA(n_components=2).fit_transform(data)
        data = StandardScaler().fit_transform(data)  # scaling the data
        data_pca = FactorAnalysis(rotation="varimax", n_components=2).fit_transform(data)

        # data_pca = MDS(n_components=2).fit_transform(data)
        # data_pca = MDS(n_components=2).fit_transform(data)

        data_complex = data_pca[:, 0] + data_pca[:, 1]*1j
        rs[dir] = np.abs(data_complex)
        angles[dir] = np.angle(data_complex)
        angles[dir] -= angles[dir][0]  # Self-Direction angle is 0
        angles[dir] += (angles[dir] < 0) * 2*np.pi
        # angles *= -1

        # correlation with the theoretical order (clockwise or counterclockwise)

        # def highest_correlation(angles, theoretical_ranks):
        #
        #     angles = np.array(angles)
        #     theoretical_ranks = np.array(theoretical_ranks)
        #     all_possible_correlations = []
        #     for i in range(len(theoretical_ranks)):
        #         all_possible_correlations.extend([
        #             compute_correlation(angles, theoretical_ranks),
        #             compute_correlation(-angles, theoretical_ranks)
        #         ])
        #
        #         if i < len(theoretical_ranks) - 1:
        #             # prepare for next iteration
        #             # rotate - (i-th is the last one, i+1-th is the first one)
        #             angles[i] += 2*np.pi
        #
        #             if angles[i+1] == 0:
        #                 # special case of conformity and tradition
        #                 angles[i+1] += 2*np.pi
        #
        #             else:
        #                 angles -= angles[i+1]
        #
        #             theoretical_ranks[i] += len(theoretical_ranks)
        #
        #             if theoretical_ranks[i+1] == 1:
        #                 # special case of conformity and tradition
        #                 theoretical_ranks[i+1] += len(theoretical_ranks)
        #             else:
        #                 theoretical_ranks -= theoretical_ranks[i+1]
        #
        #     return np.max(all_possible_correlations)

        # correlation_with_theory = highest_correlation(angles[dir], theoretical_ranks_in_order)
        # correlation_with_theory = compute_correlation(angles[dir], theoretical_ranks_in_order)

        correlation_with_theory_ccw = compute_correlation(angles[dir], theoretical_ranks_in_order)

        data_complex = -1*data_pca[:, 0] + data_pca[:, 1]*1j
        rs[dir] = np.abs(data_complex)
        angles[dir] = np.angle(data_complex)
        angles[dir] -= angles[dir][0]  # Self-Direction angle is 0
        angles[dir] += (angles[dir] < 0) * 2*np.pi
        correlation_with_theory_cw = compute_correlation(angles[dir], theoretical_ranks_in_order)

        correlation_with_theory = np.maximum(correlation_with_theory_cw, correlation_with_theory_ccw)


        per_dir_structure_correlations[dir] = correlation_with_theory

    mean_structure_correlation = np.mean(list(per_dir_structure_correlations.values()))
    print(f"Mean structure correlation: {mean_structure_correlation:.4f}.")

    if args.plot_structure:
        num_dirs = len(directories)
        num_cols = 3  # Number of columns for subplots
        num_rows = int(np.ceil((num_dirs+1) / num_cols))  # Number of rows for subplots

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
        axes = axes.flatten()  # Flatten to easily index with a single loop

        # plot theoretical structure
        ax = axes[0]

        theoretical_angles = [((r - 1) / 10) * 2 * np.pi for r in theoretical_ranks_in_order]

        theoretical_rs = [1] * len(theoretical_values_in_order)
        # equally separated angles (but conf and trad are the same angle)
        theoretical_rs[theoretical_values_in_order.index("Conformity")] = 0.8
        theoretical_rs[theoretical_values_in_order.index("Tradition")] = 1.2

        def plot_structure(ax, angles, rs, title=None, color=None, labels=None):

            # make self-direction top
            theor_complex_rot = rs * np.exp(1j * (np.array(angles) + np.pi / 2))
            coords = np.column_stack([theor_complex_rot.real, theor_complex_rot.imag])

            ax.scatter(x=coords[:, 0], y=coords[:, 1], color=color)
            for dot in coords:
                ax.plot([0, dot[0]], [0, dot[1]], color="gray", linewidth=1)

            if labels:
                for i, v in enumerate(labels):
                    ax.annotate(v, coords[i], fontsize=9)

            ax.set_title(title, y=1.05, fontsize=15)

        # plot_structure(
        #     ax=ax,
        #     angles=theoretical_angles, rs=theoretical_rs, labels=theoretical_values_in_order,
        #     title="Theoretical order", color="green"
        # )
        # plot other structures

        for i, dir in enumerate(directories):
            # idx = i + 1
            idx = i

            ax = axes[idx]
            plot_structure(
                ax=ax,
                angles=angles[dir], rs=rs[dir], labels=theoretical_values_in_order,
                title=f"{label_parser(dir)} (r={per_dir_structure_correlations[dir]:.3f})"
            )

        # Hide any empty subplots
        for j in range(idx + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.subplots_adjust(wspace=0.35, hspace=0.35, top=0.9, bottom=0.1, left=0.2, right=0.8)
        plt.suptitle(f"{dir_2_data[directories[0]]['args']['engine']} (r={mean_structure_correlation:.3f})", fontsize=20)
        plt.show()

    return mean_structure_correlation, per_dir_structure_correlations, intercorrelations

def compute_cronbach_alpha(dir_2_data, keys):

    print(colored("\n\n--------------------------------------------------", "green"))
    print(colored("Cronbach Alpha", "green"))
    print(colored("--------------------------------------------------", "green"))

    directories = list(dir_2_data.keys())

    cronbach_alphas = {}
    for dir in directories:
        cronbach_alphas[dir] = {}

        # load item indices for factors
        profile_values_idx_json = os.path.join(os.path.join(dir_2_data[dir]['args']['data_dir'], "raw"), "values.json")
        with open(profile_values_idx_json) as f:
            profile_values_idx = json.load(f)
        profile_values_idx = {k: np.array(v) - 1 for k, v in profile_values_idx.items() if k != "_comment"}

        for key in keys:

            if "pvq" in test_set_name or "svs" in test_set_name:
                _, centered_answers = per_part_normalized_scores(
                    dir_2_data, dir,
                    test_set_name, key,
                    return_all_answers=True
                )

            key_idxs = profile_values_idx[key]
            # item -> answers from all participants
            data_pd = pd.DataFrame({idx: centered_answers[:, idx] for idx in key_idxs})
            cr_alpha = pg.cronbach_alpha(data=data_pd)[0]
            cronbach_alphas[dir][key] = cr_alpha

    dir_cronbach_alphas = {d: np.mean(list(cronbach_alphas[d].values())) for d in directories}
    print("Per value alphas:")
    key_cronbach_alphas = {k: np.mean(list([dir_alphas[k] for dir_alphas in cronbach_alphas.values()])) for k in keys}
    pprint(key_cronbach_alphas)
    mean_cronbach_alpha = np.mean(list(dir_cronbach_alphas.values()))
    print(f"Mean: {mean_cronbach_alpha}")

    return mean_cronbach_alpha, cronbach_alphas

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

            if "pvq" in test_set_name or "svs" in test_set_name:
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

        "Joan of Arc",
    ]

    for key_i, key in enumerate(keys):

        # # one ref for each key (row)
        ref_key = key
        try:
            ref_dir = [d for d in key_dir_stabilities[ref_key] if "no_conv" in d][0]
        except:
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
            # reordered_names = [extract_name_brackets(n) for n in reordered_names]
            reordered_names = [n['name'] for n in reordered_names]

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


warnings.filterwarnings('error', category=ConstantInputWarning)
def compute_correlation(scores_1, scores_2):

    try:
        correlation, _ = spearmanr(scores_1, scores_2)
        return correlation

    except ConstantInputWarning:
        return np.nan

        # # # not ideal but no other way
        # if len(set(scores_1)) == 1 and len(set(scores_2)) == 1:
        #     # both constant
        #     return np.nan
        # else:
        #     # one constant  -> correlation is 0 (happens very rarely, in ipsative)
        #     return 0.0
        #     # return np.nan
        #     # print(f"Collapse setting 0.")
        #     # return np.nan



def dir_to_label(directory):

    if "format_chat_simulate_conv" in directory:
        label = extract_value(directory, "_simulate_conv_")

    elif "simulate_conv" in directory:
        label = extract_value(directory, "_simulate_conv_")

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


if __name__ == '__main__':
    import argparse

    # normalized_evaluation_data = []
    # notnorm_evaluation_data = []
    vals = []

    parser = argparse.ArgumentParser()
    parser.add_argument('directories', nargs='+', help='directories containing results.json files')
    parser.add_argument('--ips', action="store_true")
    parser.add_argument('--cfa', action="store_true")
    parser.add_argument('--structure', action="store_true")
    parser.add_argument('--plot-structure', action="store_true")
    parser.add_argument('--cronbach-alpha', action="store_true")
    parser.add_argument('--plot-matrix', action="store_true")
    parser.add_argument('--print-intercorrelations', action="store_true")
    parser.add_argument('--save-plot-matrix', action="store_true")
    parser.add_argument('--plot-save', '-ps', action="store_true")
    parser.add_argument('--no-ignore', action="store_true")
    parser.add_argument('--separate_legend', action="store_true")
    parser.add_argument('--plot-ranks', "-pr", action="store_true")
    parser.add_argument('--plot-ips', "-pi", action="store_true")
    parser.add_argument('--plot-mean', "-pm", action="store_true")
    parser.add_argument('--plot-dont-show', "-pds", action="store_true")
    parser.add_argument('--filename', type=str, default="hobbies_pvq")
    parser.add_argument('--assert-n-dirs', type=int, default=None)
    parser.add_argument('--result-json-stdout', action="store_true")
    parser.add_argument('--result-json-savepath', default=None)
    parser.add_argument('--neutral-ranks', action="store_true")
    parser.add_argument('--neutral-dir', type=str)
    parser.add_argument('--default-profile', type=str, default=None)
    parser.add_argument('--paired-dirs', nargs='+', default=None)

    args = parser.parse_args()

    if args.plot_structure:
        if not args.structure:
            print("plot_structure sets structure to True")
            args.structure = True

    if args.plot_ips:
        if not args.ips:
            print("plot_ips sets ips to True")
            args.ips = True

    if args.result_json_stdout:
        # redurect stdout to null
        import sys
        sys.stdout = open(os.devnull, 'w')

    bar_width = 0.10
    bar_margin = 1.2

    if args.plot_mean:
        fig, ax = plt.subplots(figsize=(15, 10))

    if args.no_ignore:
        ignore = []
    else:
        ignore = ["format_chat___"]

    # filter directories without results.json
    args.directories = [d for d in args.directories if os.path.isdir(d) and "results.json" in os.listdir(d)]

    args.directories = [d for d in args.directories if not any([i in d for i in ignore])]
    num_dirs = len([d for d in args.directories if os.path.isdir(d)])
    all_bars_width = num_dirs * (bar_width*bar_margin)  # bars with margins

    # chronological order
    directories = args.directories

    directories = [d for d in directories if os.path.isfile(os.path.join(d, 'results.json'))]

    print("Directories:\n\t", "\n\t".join(directories))

    if len(directories) < 2:
        raise IOError(f"Only {len(directories)} result.json files found in {args.directories}.")

    if args.assert_n_dirs and (len(directories) != args.assert_n_dirs):
        raise ValueError(f"Wrong number of dirs found {len(directories)} != {args.assert_n_dirs}.")

    dir_2_data = load_data(directories)

    test_set_name = extract_test_set_name(dir_2_data)
    test_set_values = extract_test_set_values(dir_2_data)
    keys = test_set_values

    assert directories == list(dir_2_data.keys())

    mean_rank_order_stability, key_rank_order_stabilities, key_dir_stabilities, all_ro_stabs, correlation_table = compute_rank_order_stability(dir_2_data=dir_2_data, keys=keys)

    pairwise_rank_order_stability = None

    avg_correlation_table = average_correlation_table(correlation_table)

    if args.cronbach_alpha:
        # compute cronbach alpha
        mean_cronbach_alpha, cronbach_alphas = compute_cronbach_alpha(dir_2_data=dir_2_data, keys=keys)
    else:
        mean_cronbach_alpha, cronbach_alphas = None, None

    if args.structure:
        # compute structure
        structure_correlation, structure_correlations, intercorrelations = compute_value_structure(dir_2_data=dir_2_data, keys=keys)
    else:
        structure_correlation, structure_correlations, intercorrelations = None, None, None

    if args.cfa:
        SRMRs, RMSEAs, CFIs, TLIs = conduct_cfa(dir_2_data=dir_2_data, test_set_name=test_set_name)
        CFI = np.mean(list(CFIs.values()))
        TLI = np.mean(list(TLIs.values()))
        SRMR = np.mean(list(SRMRs.values()))
        RMSEA = np.mean(list(RMSEAs.values()))

        print(colored("\n---", "green"))
        print(colored("CFA", "green"))
        print(colored("---", "green"))
        print(f"CFI: {CFI}\nTFI: {TLI}\nSRMR: {SRMR}\nRMSEA: {RMSEA}")

    else:
        SRMR, RMSEA, CFI, TLI = None, None, None, None
        SRMRs, RMSEAs, CFIs, TLIs = None, None, None, None


    if args.plot_matrix:
        model_name = get_model_name(directories)

        # split_path = directories[0].split("/")
        # assert split_path[2] == model_name
        # split_path[0] = "evaluation_results"
        # split_path = split_path[:3]
        # matrix_savepath="/".join(split_path)
        # os.makedirs(matrix_savepath, exist_ok=True)
        # matrix_savepath += "/matrix.json"

        plot_pairwise_correlations(
            correlations_matrix=avg_correlation_table,
            label_parser=label_parser,
            title=model_name,
            savepath=f"./visualizations/matrix_{model_name}.png" if args.save_plot_matrix else None
        )

    if len(keys) >= 2 and args.ips:
        mean_ipsative_stability, part_scores, ips_part_stabilities, ips_part_dir_stabilities, all_ips_corrs = compute_ipsative_stability(dir_2_data=dir_2_data, keys=keys)
    else:
        all_ips_corrs = np.nan
        mean_ipsative_stability = np.nan
        all_corrs = np.nan

    if args.default_profile:
        dir_2_data_paired = load_data([args.default_profile])
        assert len(dir_2_data_paired.keys()) == 1
        dir_neut_prof = list(dir_2_data_paired.keys())[0]

        # per participant normalize the neutral profile
        normalized_scores = []
        for key in keys:

            scores_neut_prof = np.array([d[test_set_name][key] for d in dir_2_data_paired[dir_neut_prof]["per_simulated_participant_metrics"]])

            if "pvq" in test_set_name or "svs" in test_set_name:
                # extract per participant average answer
                average_part_answer_neut_prof = np.array([
                    np.array(d[test_set_name])[:, 1].astype(float).mean() for d in dir_2_data_paired[dir_neut_prof]["answers"]
                ])
                scores_neut_prof_ = scores_neut_prof - average_part_answer_neut_prof
                scores_neut_prof = per_part_normalized_scores(dir_2_data_paired, dir_neut_prof, test_set_name, key)
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
    all_neutral_ro_stabs = None

    if args.neutral_ranks:
        print()
        print(colored("------------------------", "green"))
        print(colored("Neutral Rank-Order stability", "green"))
        print(colored("------------------------", "green"))

        # find the neutral context dir

        neutral_dir = glob.glob(args.neutral_dir + "/*/*chat___*")[0]
        dir_2_data_neutral = load_data([neutral_dir])

        # assert no topic
        assert dir_2_data_neutral[neutral_dir]['args']['simulated_conversation_theme'] is None
        assert dir_2_data_neutral[neutral_dir]['args']['simulate_conversation_theme'] is None

        values_names = [
            "Benevolence", "Universalism", "Power", "Achievement", "Tradition",
            "Conformity", "Security", "Self-Direction", "Stimulation", "Hedonism"
        ]

        mean_stability = []
        all_neutral_ro_stabs = {}
        for value in values_names:

            stab_, _, _, all_neutral_ro_stabs_val = compute_paired_rank_order_stability(
                dir_2_data, dir_2_data_neutral,
                value, value,
                "pvq_auto", "pvq_auto",
                directories_1=dir_2_data.keys(), directories_2=[neutral_dir]*len(dir_2_data)
            )

            mean_stability.append(stab_)

            all_neutral_ro_stabs[value] = all_neutral_ro_stabs_val

        neutral_rank_order_stability = np.mean(mean_stability)

        print(f"Mean stability w neutral rank: {neutral_rank_order_stability}")

    proxy_stability = None
    all_proxy_stabs = None

    if args.paired_dirs:

        print("\n\nPaired Rank-Order stability\n")
        args.paired_dirs = [d for d in args.paired_dirs if "results.json" in os.listdir(d)]

        dir_2_data_paired = load_data(args.paired_dirs)

        if dir_2_data_paired == {}:
            raise ValueError(f"No data found for paired_dirs {args.paired_dirs}")

        values_names = ["Benevolence", "Universalism", "Power", "Achievement", "Tradition", "Conformity", "Security",
                        "Self-Direction", "Stimulation", "Hedonism"]

        proxy_stability = {}
        all_proxy_stabs = {}
        for value in values_names:
            mean_stability = []
            all_proxy_stabs[value] = {}
            for race in ["elves", 'dwarves', 'orcs', 'humans', 'hobbits']:
                mean_paired_rank_order_stability, _, _, all_proxy_stabs_ = compute_paired_rank_order_stability(
                    dir_2_data, dir_2_data_paired,
                    value, f"Donation {race}",
                    "pvq_auto", "tolkien_donation"
                )
                all_proxy_stabs[value][race] = all_proxy_stabs_
                # print(f"{race}: {mean_paired_rank_order_stability}")
                mean_stability.append(mean_paired_rank_order_stability)

            proxy_stability[value] = np.mean(mean_stability)

            print(f"Mean {value} stability: {proxy_stability[value]}.")


    print()
    print(colored("------------------------", "green"))
    print(colored("Aggregated metrics", "green"))
    print(colored("------------------------", "green"))

    print("Rank-Order\tIpsative")
    print(f"{mean_rank_order_stability:.4f}\t\t{mean_ipsative_stability:.4f}")

    if args.result_json_stdout or args.result_json_savepath:
        outputs = {
            "Rank-Order": mean_rank_order_stability,
            "Pairwise_Rank-Order": avg_correlation_table,
            "Per_value_Pairwise_Rank-Order": correlation_table,
            "All_Rank-Order_stabilities": all_ro_stabs,
            "Ipsative": mean_ipsative_stability,
            "All_Ipsative_corrs": all_ips_corrs,
            "Ipsative_default_profile": mean_ipsative_stability_default_profile,
            "All_Ipsative_corrs_default_profile": all_ips_corrs_default_profile,
            "Proxy_stability": proxy_stability,
            "All_Proxy_stabilities": all_proxy_stabs,
            "Neutral_Rank-Order": neutral_rank_order_stability,
            "All_Neutral_Rank-Order_stabilities": all_neutral_ro_stabs,
            "dir_params": {d: dir_2_data[d]['params'] for d in args.directories},
            "Cronbach_alpha": mean_cronbach_alpha,
            "Cronbach_alphas": cronbach_alphas,
            "Structure_correlation": structure_correlation,
            "Structure_correlations": structure_correlations,
            "Intercorrelations": intercorrelations,
            "CFI": CFI, "CFIs": CFIs,
            "TLI": TLI, "TLIs": TLIs,
            "SRMR": SRMR, "SRMRs": SRMRs,
            "RMSEA": RMSEA, "RMSEAs": RMSEAs,
        }

        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return json.JSONEncoder.default(self, obj)

        if args.result_json_stdout:
            sys.stdout = sys.__stdout__
            print(json.dumps(outputs, cls=NumpyEncoder))

        elif args.result_json_savepath:
            with open(args.result_json_savepath, 'w') as json_file:
                json.dump(outputs, json_file, cls=NumpyEncoder, indent=4)
            print(f"Saved to {args.result_json_savepath}.")

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
                plt.savefig(savepath, dpi=600)

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
