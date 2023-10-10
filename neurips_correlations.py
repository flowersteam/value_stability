# to be run after nerurips_correlations.sh
import matplotlib.pyplot as plt
from collections import defaultdict
import csv
import numpy as np
import pandas as pd

import numpy as np

fontsize = 18
legend = False
max_x = 0

def flat_and_remove_nan(d):
    d=d.flatten()
    d=d[~np.isnan(d)]
    return d

plot_data = []
cols = []
# variance data
filenames = ['corr_variance.csv', 'corr_controllability.csv']

questionnaire_dict = {
    # "PVQ": "o",
    # "Hof": "o", #"x",
    # "Big5": "o", #"d"
}

questionnaire_legends = defaultdict(list)

for questionnaire in questionnaire_dict.keys():
    model_data = defaultdict(list)

    for filename in filenames:

        data_pd = pd.read_csv(filename, header=None)

        for ind, row in data_pd.iterrows():
            model_name = row[0].lstrip().rstrip()

            quest_ind = {
                "PVQ": 0,
                "Hof": 1,
                "Big5": 2,
            }
            start_ind = 1+4*quest_ind[questionnaire]
            end_ind = start_ind + 4

            data_row_pf = row[start_ind:end_ind]
            # data_row_pf = row[1:]  # all questionnaires at once

            data = pd.to_numeric(data_row_pf, errors='coerce').to_numpy(float)
            data = flat_and_remove_nan(data)

            if "variance" in filename:
                data /= 10**3

            model_data[model_name].append(data)


    colors_dict = {
        # "GPT-3.5": "green",
        # "OA": "blue",
        # "StVicuna": "red",
        # "StLM": "pink",
        "" : ""
    }

    for model_name, data in model_data.items():
        max_x = max([max_x, max(data[0])])
        scatter_plot = plt.scatter(*data, label=model_name, c=colors_dict.get(model_name, "black"), marker=questionnaire_dict[questionnaire])
        if legend:
            questionnaire_legends[questionnaire].append(scatter_plot)

if legend:
    for i, (questionnaire, legend_handles) in enumerate(questionnaire_legends.items()):
        leg = plt.legend(handles=legend_handles, loc="upper right", bbox_to_anchor=(1, 1-0.3*i), fontsize=fontsize)
        plt.gca().add_artist(leg)

plt.ylabel("Correspondence", fontsize=fontsize)
plt.xlabel("Permutation Variance", fontsize=fontsize)

plt.tick_params(axis='both', which='major', labelsize=fontsize)

# plt.ylim(-0.05, 0.8)
# xmax={
#     "PVQ": "0.12",
#     "Hof": "0.5",  # "x",
#     "Big5": "0.5", #"d"
# }
# plt.xlim(0, 0.12)

# plt.show()
quest_names = "_".join(questionnaire_dict.keys())
filename =f"visualizations/neurips_plots/coh_var_{quest_names}.png"
plt.savefig(filename)
print(f"saved to {filename}")
# plt.show()
