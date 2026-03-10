from itertools import combinations
from benchbench.data import load_ordinal_benchmark
from benchbench.measures import cardinal, ordinal
from benchbench.utils.base import rankdata
from benchbench.utils.win_rate import WinningRate
from benchbench.utils.metric import get_rank_diff

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

chunk_labels = [
    'chunk_0',
    'chunk_1',
    'chunk_2',
    'chunk_3',
    'chunk_4',
    'chunk_chess_0',
    'chunk_grammar_1',
    'chunk_no_conv',
    'chunk_svs_no_conv'
]

def model_2_color(model):
    model_lower = str(model).lower()
    if "llama" in model_lower:
        return "blue"
    elif "mixtral" in model_lower:
        return "orange"
    elif "mistral" in model_lower or "zephyr" in model_lower:
        return "orange"
    elif "phi" in model_lower:
        return "red"
    elif "qwen" in model_lower:
        return "purple"
    elif "gpt" in model_lower:
        return "green"
    elif "command" in model_lower:
        return "orchid"
    else:
        return "black"

def extract_chunk_label(model, dir, assert_chunk_labels=None):
    label = dir.removeprefix(f"./Leaderboard/results/stability_leaderboard/{model}/").split("_202")[0]
    if assert_chunk_labels is not None:
        assert label in assert_chunk_labels, f"Label {label} not recognized."

    return label


def parse_pairwise_staiblity_dirs_to_labels(pairwise_stabilities, assert_chunk_labels=None):
    parsed_pairwise_stabilities = {}

    for model, model_pairwise_staiblities in pairwise_stabilities.items():
        parsed_pairwise_stabilities[model] = {}

        for d1, values in model_pairwise_staiblities.items():
            l1 = extract_chunk_label(model=model, dir=d1, assert_chunk_labels=assert_chunk_labels)

            parsed_pairwise_stabilities[model][l1] = {}

            for d2, value in values.items():
                l2 = extract_chunk_label(model=model, dir=d2, assert_chunk_labels=assert_chunk_labels)

                parsed_pairwise_stabilities[model][l1][l2] = value

    return parsed_pairwise_stabilities


def compute_stability_and_sensitivity(data_df=None, pairwise_stabilities=None, type="cardinal", cols=None):
    assert type in ["cardinal", "ordinal"]
    # Compute sensitivity and stability

    # if pairwise_stabilities is not None:
    #     data, cols = {}, []
    #     models = pairwise_stabilities.keys()
    #
    #     for ch1, ch2 in combinations(chunk_labels, 2):
    #         col = f"{ch1}-{ch2}"
    #         cols.append(col)
    #         data[col] = []
    #         for model in models:
    #             data[col].append(pairwise_stabilities[model][ch1][ch2])
    #
    #     data_df = pd.DataFrame.from_dict(data)

    if cols is None:
        cols = data_df.columns


    if type == "cardinal":
        diversity_w, diversity_max_MRC = cardinal.get_diversity(data_df, cols)
        sensitivity_tau, sensitivity_MRC = cardinal.get_sensitivity(data_df, cols)

    elif type == "ordinal":

        # check if rank diff is too big; not clear what this does?
        # what are the assumptions on the data_df order of models?
        # wr = WinningRate(data_df, cols)
        # new_rank = rankdata(-wr.get_winning_rate())
        # old_rank = np.arange(len(data_df))
        # rank_diff = get_rank_diff(new_rank, old_rank)

        # if rank_diff[0] > 0.1:
        #     print("Rank diff too big")
        #     diversity_w, diversity_max_MRC = np.nan, np.nan
        #     sensitivity_tau, sensitivity_MRC = np.nan, np.nan
        # else:

        diversity_w, diversity_max_MRC = ordinal.get_diversity(data_df, cols)

        # inv_indices = None
        # inv_indices = list(np.arange(np.maximum(len(data_df) // 4, 6)))
        # print(ordinal.get_sensitivity(data_df, cols, inv_indices=inv_indices)[1])
        # sensitivity_tau, sensitivity_MRC = ordinal.get_sensitivity(data_df, cols, inv_indices=inv_indices)
        sensitivity_tau, sensitivity_MRC = ordinal.get_sensitivity(data_df, cols)


    else:
        raise ValueError(f"Unknown type: {type}")

    return diversity_w, diversity_max_MRC, sensitivity_tau, sensitivity_MRC

def compute_win_rates(data_df):
    win_rates_df = pd.DataFrame(np.zeros(data_df.shape), columns=data_df.columns)


    # compute ranks
    for model_i in range(data_df.shape[0]):
        for metric_j in range(data_df.shape[1]):
            col = data_df.columns[metric_j]
            count = np.sum(data_df[col] < data_df.at[model_i, col]) - (
                    data_df.at[model_i, col] < data_df.at[model_i, col])
            win_rates_df.at[model_i, col] = count / (data_df.shape[0] - 1)

    return win_rates_df

def plot_ranked_models(model_scores, title=None, y_label=None, savepath=None):
    # PLOS
    xs, ys = zip(*model_scores.items())
    fig, ax = plt.subplots(figsize=(10, 5))


    # Plot
    colors = list(map(model_2_color, xs))

    ax.bar(xs, ys, color=colors)
    ax.set_xticklabels(xs, rotation=90)
    plt.ylim((-0.1, 1))
    plt.ylabel(y_label)
    plt.subplots_adjust(bottom=0.43)

    title_str = f"{title} " if title is not None else ""


    if title_str != "":
        plt.title(title_str)

    if savepath:
        plt.savefig(savepath)
        print(f"Saved to: {savepath}")
    else:
        plt.show()
