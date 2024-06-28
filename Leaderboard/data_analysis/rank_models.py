import json

import pandas as pd

from plot_utils import *

models = [
    "phi-3-mini-128k-instruct",
    "phi-3-medium-128k-instruct",
    "Mistral-7B-Instruct-v0.1",
    "Mistral-7B-Instruct-v0.2",
    "Mistral-7B-Instruct-v0.3",
    "Mixtral-8x7B-Instruct-v0.1",
    "Mixtral-8x22B-Instruct-v0.1",
    "command_r_plus",
    "llama_3_8b_instruct",
    "llama_3_70b_instruct",
    "Qwen2-7B-Instruct",
    "Qwen2-72B-Instruct",
    "gpt-3.5-turbo-0125",
    "gpt-4o-0513"
]

mmlu_models = {
    "phi-3-mini-128k-instruct": 68.1,
    "phi-3-medium-128k-instruct": 76.6,
    "Mistral-7B-Instruct-v0.1": 55.38,
    "Mistral-7B-Instruct-v0.2":  60.78,
    "Mistral-7B-Instruct-v0.3": ...,
    "Mixtral-8x7B-Instruct-v0.1": 71.16,
    "Mixtral-8x22B-Instruct-v0.1": 77.77,
    "command_r_plus": 75.73,
    "llama_3_8b_instruct": 67.07,
    "llama_3_70b_instruct": 80.06,
    "Qwen2-7B-Instruct": 70.5,
    "Qwen2-72B-Instruct": 82.3,
    "gpt-3.5-turbo-0125": 70.00,  # old version - technical report
    "gpt-4o-0513": 88.7,
}


ANALYSIS_RESULTS_DIR = "./Leaderboard/data_analysis/analysis_results"

# validation_metrics = ["Structure_correlations", "CFIs", "TLIs", "SRMRs", "RMSEAs"]
validation_metrics = ["Structure_correlations", "CFIs", "SRMRs", "RMSEAs"]

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

    # Compute sensitivity and stability
    data, cols = {}, []
    for ch1, ch2 in combinations(chunk_labels, 2):
        col = f"RO-{ch1}-{ch2}"
        cols.append(col)
        data[col] = []
        for model in models:
            data[col].append(pairwise_stabilities[model][ch1][ch2])

    for validation_metric in validation_metrics:

        for model in models:
            results[model][validation_metric] = {
                extract_chunk_label(model, k, chunk_labels): v for k, v in results[model][validation_metric].items()
            }

        def parse_validation_metric_name(validation_metric):
            if validation_metric in ["SRMRs", "RMSEAs"]:
                validation_metric = f"inv{validation_metric}"

            return validation_metric.rstrip("s")

        for ch in chunk_labels:
            col = f"{parse_validation_metric_name(validation_metric)}-{ch}"
            cols.append(col)
            data[col] = []
            for model in models:
                if validation_metric in ["Structure_correlations", "CFIs", "TLIs"]:
                    data[col].append(results[model][validation_metric][ch])
                elif validation_metric in ["SRMRs", "RMSEAs"]:
                    data[col].append(1-results[model][validation_metric][ch])
                else:
                    raise ValueError(f"Metric {validation_metric} not defined.")

    # Reliability: Cronbach's alpha
    import numpy as np
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

    cardinal_df = pd.DataFrame.from_dict(data)
    cardinal_scores_series = cardinal_df.mean(axis=1)
    cardinal_scores_df = pd.DataFrame(cardinal_scores_series, columns=["Score (Cardinal)"])

    ordinal_df = compute_win_rates(cardinal_df)
    ordinal_scores_series = ordinal_df.mean(axis=1)
    ordinal_scores_df = pd.DataFrame(ordinal_scores_series, columns=["Win rate (Ordinal)"])

    from math import comb
    assert cardinal_df.shape[1] == comb(len(chunk_labels), 2) + (len(validation_metrics) + 1) * len(chunk_labels)

    for data_df, final_scores_series, type in zip(
            [cardinal_df, ordinal_df],
            [cardinal_scores_series, ordinal_scores_series],
            ["cardinal", "ordinal"]
    ):
        # compute stability and sensitivity
        diversity_w, _, _, sensitivity_MRC = compute_stability_and_sensitivity(data_df=data_df, type=type)

        print(f"Diversity: {diversity_w:.2f} W; Sensitivity: {sensitivity_MRC:.2f} (MRC)")

        final_scores = dict(zip(models, final_scores_series))
        final_scores = dict(sorted(final_scores.items(), key=lambda x: x[1], reverse=True))

        print("Scores")
        for i, (m, v) in enumerate(final_scores.items()):
            print(f"{i + 1}. {m} - {v:.2f}")

        plot_ranked_models(final_scores, diversity_w, sensitivity_MRC, title=f"{type} Leaderboard")
        

# create leaderboard dataframe

# Define which columns to average
leaderboard_metrics = {
    "Rank-Order Stability": [f"RO-{ch1}-{ch2}" for ch1, ch2 in combinations(chunk_labels, 2)],
    "Rank-Order Stability (internal)": [f"RO-{ch1}-{ch2}" for ch1, ch2 in combinations([c for c in chunk_labels if "svs" not in c], 2)],
    "Rank-Order Stability (external)": [f"RO-{ch1}-{ch2}" for ch1, ch2 in combinations([c for c in chunk_labels if "svs" in c], 2)],
    **{validation_metric: [
        f"{parse_validation_metric_name(validation_metric)}-{ch}" for ch in chunk_labels
    ] for validation_metric in validation_metrics},
    "Cronbach_alpha": [f"Cronbach_alpha-{ch}" for ch in chunk_labels]
}

leaderboard_data = {}
for leaderboard_metric, metrics_list in leaderboard_metrics.items():
    # Calculate the averages
    leaderboard_data[leaderboard_metric] = data_df[metrics_list].mean(axis=1)

# Create the new DataFrame with averaged columns
leaderboard_df = pd.DataFrame(leaderboard_data)
leaderboard_df = pd.concat([cardinal_scores_df, ordinal_scores_df, leaderboard_df], axis=1)
leaderboard_df.index = models

leaderboard_save_path = ANALYSIS_RESULTS_DIR+"/leaderboard.csv"
leaderboard_df.to_csv(leaderboard_save_path)
print(leaderboard_df)
print(f"Saved to: {leaderboard_save_path}")




