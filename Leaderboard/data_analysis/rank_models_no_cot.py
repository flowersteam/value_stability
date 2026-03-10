import json
import numpy as np


from plot_utils import *


models = [
    # "GLM-4-32B-0414",
    #
    # "Qwen3-235B-A22B-FP8",
    # "Qwen3-32B-A3B",
    # "Qwen3-32B",
    # "Qwen3-8B",
    # "Qwen3-4B",
    #
    # "reka-flash-3",
    #
    # "DeepSeek-V3-0324",
    # # "DeepSeek-V3-0324_user",
    #
    # "gemma-3-27b-it",
    #
    # "Llama-4-Scout-17B-16E-Instruct",

    "Llama-3.3-70B-Instruct",
    "Llama-3.1-70B-Instruct",
    "Llama-3.1-Nemotron-70B-Instruct",
    "Llama-3.1-8B-Instruct",

    "Llama-3.2-3B-Instruct",
    "Llama-3.2-1B-Instruct",

    "Mistral-Large-Instruct-2411",
    "Mistral-Large-Instruct-2407",
    "Mistral-Nemo-Instruct-2407",
    "Mistral-Small-3.1-24B-Instruct-2503",

    "Mistral-7B-Instruct-v0.2",
    "Mixtral-8x7B-Instruct-v0.1",

    # "QwQ-32B",

    "Qwen2.5-VL-72B-Instruct",
    "Qwen2.5-VL-7B-Instruct",
    "Qwen2.5-VL-3B-Instruct",
    # "Qwen2.5-72B-Instruct",
    # "Qwen2.5-32B-Instruct",
    "Qwen2.5-14B-Instruct-1M",

    "phi-4", # not enough context for CoT 12k and generates more
    "phi-3-medium-128k-instruct",

    # "shuttle-3.5"

    "Dracarys2-72B-Instruct",

    "Nautilus-70B-v0.1",
    "Cydonia-22B-v1.2",
    "Ministrations-8B-v1",

    "dummy"
]

ANALYSIS_RESULTS_DIR = "./Leaderboard/data_analysis/analysis_results_no_cot"

# metrics for which 0 is optimal
# decreasing_metrics = ["All_Stress", "All_Rank_distance", "All_SRMR", "All_RMSEA"]
# validation_metrics = ["All_Stress", "All_Rank_distance", "All_Separability", "All_CFI", "All_SRMR", "All_RMSEA"]

decreasing_metrics = ["All_Stress", "All_SRMR", "All_RMSEA"]

validation_metrics = ["All_Stress", "All_CFI", "All_SRMR", "All_RMSEA"]
# validation_metrics = ["All_Stress", "All_CFI"]

cfa_metrics = ["All_CFI", "All_SRMR", "All_RMSEA"]
cronbach_alpha = False

save_plot = True

def to_singular(str):
    return str.removeprefix("All_")

def parse_metric_name(metric, decreasing_metrics):
    if metric in decreasing_metrics:
        metric = f"1 - {metric}"

    return to_singular(metric)


if __name__ == '__main__':
    results = {}
    pairwise_stabilities = {}

    for model in models:
        with open(ANALYSIS_RESULTS_DIR+"/"+model+".json") as f:
            model_data = json.load(f)
            results[model] = model_data
            pairwise_stabilities[model] = model_data['Pairwise_Rank-Order']

    # parse dir names to chunk_labels
    pairwise_stabilities = parse_pairwise_staiblity_dirs_to_labels(
        pairwise_stabilities=pairwise_stabilities,
        assert_chunk_labels=chunk_labels
    )

    # Add pairwise RO metrics
    data, cols = {}, []
    for ch1, ch2 in combinations(chunk_labels, 2):
        col = f"RO-{ch1}-{ch2}"
        cols.append(col)
        data[col] = []
        for model in models:
            data[col].append(pairwise_stabilities[model][ch1][ch2])

    # add validation metrics (invert decreasing ones)
    for validation_metric in validation_metrics:

        for model in models:
            results[model][validation_metric] = {
                extract_chunk_label(model, k, chunk_labels): v for k, v in results[model][validation_metric].items()
            }

        for ch in chunk_labels:
            col = f"{parse_metric_name(validation_metric, decreasing_metrics)}-{ch}"
            cols.append(col)
            data[col] = []
            for model in models:
                if validation_metric in cfa_metrics:
                    # magnifying glass CFA (one model per per high level values)
                    assert len(results[model][validation_metric][ch]) == 4
                    metric_score = np.mean(list(results[model][validation_metric][ch].values()))
                else:
                    metric_score = results[model][validation_metric][ch]

                if validation_metric in decreasing_metrics:
                    data[col].append(1-metric_score)
                else:
                    data[col].append(metric_score)

    if cronbach_alpha:
        # add Cronbach's alpha
        for model in models:
            results[model]["Cronbach_alphas"] = {
                extract_chunk_label(model, k, chunk_labels): v for k, v in results[model]["Cronbach_alphas"].items()
            }

        for ch in chunk_labels:
            col = f"Cronbach_alpha-{ch}"
            cols.append(col)
            data[col] = []
            for model in models:
                data[col].append(np.mean(list(results[model]["Cronbach_alphas"][ch].values())))

    data_df = pd.DataFrame.from_dict(data)
    assert len(data_df.columns) == 72

    # cardinal
    cardinal_df = pd.DataFrame.from_dict(data)
    cardinal_scores_series = cardinal_df.mean(axis=1)
    cardinal_scores_df = pd.DataFrame(cardinal_scores_series, columns=["Cardinal (Score)"])

    # ordinal
    ordinal_df = compute_win_rates(data_df)
    ordinal_scores_series = ordinal_df.mean(axis=1)
    ordinal_scores_df = pd.DataFrame(ordinal_scores_series, columns=["Ordinal (Win rate)"])

    #  WinningRate(data_df, data_df.columns).get_winning_rate()

    from math import comb
    assert cardinal_df.shape[1] == comb(len(chunk_labels), 2) + (len(validation_metrics) + int(cronbach_alpha)) * len(chunk_labels)

    # compute stability and sensitivity and plot model ranks
    for agg_data_df, final_scores_series, type in zip(
            [cardinal_df, ordinal_df],
            [cardinal_scores_series, ordinal_scores_series],
            ["cardinal", "ordinal"]
    ):
        print(f"Type: {type}")
        # compute stability and sensitivity
        diversity_w, _, _, sensitivity_MRC = compute_stability_and_sensitivity(data_df=data_df, type=type)

        print(f"Diversity: {diversity_w:.2f} (W); Sensitivity: {sensitivity_MRC:.2f} (MRC)")

        final_scores = dict(zip(models, final_scores_series))
        final_scores = dict(sorted(final_scores.items(), key=lambda x: x[1], reverse=True))

        print("Scores")
        for i, (m, v) in enumerate(final_scores.items()):
            print(f"{i + 1}. {m} - {v:.2f}")

        if save_plot:
            savepath = f"./visualizations/{type}.svg"
        else:
            savepath = None

        title_str = f"{type.capitalize()} Leaderboard"
        title_str += f" - diversity: {diversity_w:.2f} (W), sensitivity: {sensitivity_MRC:.2f} (MRC)"

        plot_ranked_models(
            final_scores,
            title=title_str,
            y_label="Win rate" if type == "ordinal" else "Mean score",
            savepath=savepath
        )
        

# Create Leaderboard dataframe
##############################

# Define which columns to average
leaderboard_metrics = {
    "Rank-Order Stability": [f"RO-{ch1}-{ch2}" for ch1, ch2 in combinations(chunk_labels, 2)],
    **{parse_metric_name(validation_metric, decreasing_metrics): [
        f"{parse_metric_name(validation_metric, decreasing_metrics)}-{ch}" for ch in chunk_labels
    ] for validation_metric in validation_metrics}
}
if cronbach_alpha:
    leaderboard_metrics = {
        **leaderboard_metrics, **{
            "Cronbach alpha": [f"Cronbach_alpha-{ch}" for ch in chunk_labels]
        }
    }

leaderboard_data = {}
for leaderboard_metric, metrics_list in leaderboard_metrics.items():
    # Calculate the averages
    leaderboard_data[leaderboard_metric] = data_df[metrics_list].mean(axis=1)


# Create the new DataFrame with averaged columns
leaderboard_df = pd.DataFrame(leaderboard_data)

# sens and div on a per-metric basis
# diversity_w, _, _, sensitivity_MRC = compute_stability_and_sensitivity(data_df=leaderboard_df, type="ordinal")
leaderboard_df = pd.concat([ordinal_scores_df, cardinal_scores_df, leaderboard_df], axis=1)
leaderboard_df.index = models



# Revert decreasing metrics: metric = 1-metric

# rename decreasing metric columns, e.g. 1-SRMRs -> SRMR
columns_rename_dict = {
    parse_metric_name(dec_metric, decreasing_metrics): to_singular(dec_metric) for dec_metric in decreasing_metrics
}
leaderboard_df = leaderboard_df.rename(columns=columns_rename_dict)
# invert values
for column_name in columns_rename_dict.values():
    leaderboard_df[column_name] = 1-leaderboard_df[column_name]



# rename the rest of the metrics
leaderboard_df = leaderboard_df.rename(columns={
    "Rank-Order Stability": "RO Stability",
    "Rank_distance": "Rank Distance",
})
leaderboard_save_path = ANALYSIS_RESULTS_DIR+"/leaderboard.csv"
leaderboard_df.to_csv(leaderboard_save_path, index_label="Model")


pd.set_option("display.max_columns", 4)  # Show all columns
pd.set_option("display.width", 200)  # Increase output width
# print(leaderboard_df.sort_values(by="Ordinal (Win rate)", ascending=False))
print(leaderboard_df.sort_values(by="RO Stability", ascending=False))

print(f"Saved to: {leaderboard_save_path}")