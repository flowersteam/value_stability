#! python3

from pathlib import Path
import json
from collections import defaultdict
import scipy.stats as stats
from termcolor import colored

# Ipsative
# mean, STD, n
ipsative_human_change = 0.59, 0.25, 270

# Simulated conversations
data_sim_conv = {}
data_sim_conv["chess_grammar"] = (0.78, 0.12, 50)
data_sim_conv["chess_history"] = (0.70, 0.19, 50)
data_sim_conv["chess_joke"] = (0.48, 0.27, 50)
data_sim_conv["chess_poem"] = (0.87, 0.07, 50)
data_sim_conv["grammar_history"] = (0.70, 0.18, 50)
data_sim_conv["grammar_joke"] = (0.40, 0.32, 50)
data_sim_conv["grammar_poem"] = (0.90, 0.06, 50)
data_sim_conv["history_joke"] = (0.52, 0.30, 50)
data_sim_conv["history_poem"] = (0.74, 0.20, 50)
data_sim_conv["joke_poem"] = (0.42, 0.32, 50)

# Textual formats
data_text_format = {}
data_text_format["chat_code_cpp"] = (0.05, 0.23, 50)
data_text_format["chat_code_py"] = (0.31, 0.28, 50)
data_text_format["chat_conf_toml"] = (0.86, 0.06, 50)
data_text_format["chat_latex"] = (0.68, 0.19, 50)
data_text_format["code_cpp_code_py"] = (0.30, 0.35, 50)
data_text_format["code_cpp_conf_toml"] = (0.08, 0.24, 50)
data_text_format["code_cpp_latex"] = (0.20, 0.32, 50)
data_text_format["code_py_conf_toml"] = (0.33, 0.31, 50)
data_text_format["code_py_latex"] = (0.55, 0.27, 50)
data_text_format["conf_toml_latex"] = (0.68, 0.23, 50)

# Wikipedia articles
data_wiki = {}
data_wiki["classical_gospel"] = (0.84, 0.09, 50)
data_wiki["classical_heavy_metal"] = (0.80, 0.10, 50)
data_wiki["classical_hip_hop"] = (0.86, 0.07, 50)
data_wiki["classical_jazz"] = (0.82, 0.11, 50)
data_wiki["classical_reggae"] = (0.84, 0.09, 50)
data_wiki["gospel_heavy_metal"] = (0.77, 0.09, 50)
data_wiki["gospel_hip_hop"] = (0.83, 0.08, 50)
data_wiki["gospel_jazz"] = (0.79, 0.10, 50)
data_wiki["gospel_reggae"] = (0.88, 0.06, 50)
data_wiki["heavy_metal_hip_hop"] = (0.91, 0.05, 50)
data_wiki["heavy_metal_jazz"] = (0.92, 0.05, 50)
data_wiki["heavy_metal_reggae"] = (0.86, 0.09, 50)
data_wiki["hip_hop_jazz"] = (0.90, 0.05, 50)
data_wiki["hip_hop_reggae"] = (0.88, 0.08, 50)
data_wiki["jazz_reggae"] = (0.89, 0.06, 50)

human_mean, human_std, human_nobs = ipsative_human_change
p_limit = 0.05

for data in [data_sim_conv, data_text_format, data_wiki]:
    # we only compare those with llm correlation < human correlation
    data = {key: value for key, value in data.items() if value[0] < human_mean}

    if len(data) == 0:
        continue

    # bonferroni correction
    p_limit_bonf = p_limit / len(data)

    print("------------")
    for key, value in data.items():
        llm_mean, llm_std, llm_nobs = value
        pvalue = stats.ttest_ind_from_stats(
            mean1=human_mean,
            std1=human_std,
            nobs1=human_nobs,
            mean2=llm_mean,
            std2=llm_std,
            nobs2=llm_nobs,
        ).pvalue

        if pvalue < p_limit_bonf:
            print(colored(f"{key} - Mean: {llm_mean} p={pvalue:.5f}", "green"))
        else:
            print(f"{key} - Mean: {llm_mean} p={pvalue:.5f}")

