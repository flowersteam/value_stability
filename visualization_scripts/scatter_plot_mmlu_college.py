import matplotlib.pyplot as plt
import json


def load_from_json(path, subjs, subj_for_macro_avg=None):
    with open(path, "r") as read_file:
        accs_json = json.load(read_file)

    subject_accs = [accs_json[s] for s in subjs]

    if subj_for_macro_avg is not None:
        macro_avg = sum([accs_json[s] for s in subj_for_macro_avg])/len(subj_for_macro_avg)
        subject_accs += [macro_avg]

    return subject_accs


subj_to_label = {
    "tomi_first_order_no_tom": "First_order TB",  # no tom
    "tomi_first_order_tom": "First_order FB",  # tom
    "tomi_memory": "Memory",
    "tomi_reality": "Reality",
    "tomi_second_order_no_tom": "Second_order(TB)",  # no tom
    "tomi_second_order_tom": "Second_order(FB)",  # tom
    "average": "Avg (micro)"
}

subjs = [
    # "medical_genetics",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
]
subj_for_macro_avg = subjs

title = ""
data = [
    # x-axis
    # [subj_to_label[s] for s in subjs] + ["Avg (macro)"],
    subjs + ["Avg (macro)"],

    # LLama -exp
    # [
    #     ("default", load_from_json("results/mmlu_college_exp_def_llama_30B_data_ntrain_0/results.json", subjs, subj_for_macro_avg=subj_for_macro_avg)),  # <- rerun wo genetics
    #     ("bio expert", load_from_json("results/mmlu_college_exp_bio_llama_30B_data_ntrain_0/results.json", subjs, subj_for_macro_avg=subj_for_macro_avg)),  # <- rerun wo genetics
    #     ("chem expert", load_from_json("results/mmlu_college_exp_chem_llama_30B_data_ntrain_0/results.json", subjs, subj_for_macro_avg=subj_for_macro_avg)),
    #     ("cs expert", load_from_json("results/mmlu_college_exp_cs_llama_30B_data_ntrain_0/results.json", subjs, subj_for_macro_avg=subj_for_macro_avg)),
    #     ("math expert", load_from_json("results/mmlu_college_exp_math_llama_30B_data_ntrain_0/results.json", subjs, subj_for_macro_avg=subj_for_macro_avg)),
    #     ("med expert", load_from_json("results/mmlu_college_exp_med_llama_30B_data_ntrain_0/results.json", subjs, subj_for_macro_avg=subj_for_macro_avg)),
    #     ("phy expert", load_from_json("results/mmlu_college_exp_phy_llama_30B_data_ntrain_0/results.json", subjs, subj_for_macro_avg=subj_for_macro_avg)),
    # ],
    # LLama - prof
    [
        ("default", load_from_json("results/mmlu_college_prof_def_llama_30B_data_ntrain_0/results.json", subjs, subj_for_macro_avg=subj_for_macro_avg)),  # <- rerun wo genetics
        ("bio prof", load_from_json("results/mmlu_college_prof_bio_llama_30B_data_ntrain_0/results.json", subjs, subj_for_macro_avg=subj_for_macro_avg)),  # <- rerun wo genetics
        ("chem prof", load_from_json("results/mmlu_college_prof_chem_llama_30B_data_ntrain_0/results.json", subjs, subj_for_macro_avg=subj_for_macro_avg)),
        ("cs prof", load_from_json("results/mmlu_college_prof_cs_llama_30B_data_ntrain_0/results.json", subjs, subj_for_macro_avg=subj_for_macro_avg)),
        ("math prof", load_from_json("results/mmlu_college_prof_math_llama_30B_data_ntrain_0/results.json", subjs, subj_for_macro_avg=subj_for_macro_avg)),
        ("med prof", load_from_json("results/mmlu_college_prof_med_llama_30B_data_ntrain_0/results.json", subjs, subj_for_macro_avg=subj_for_macro_avg)),
        ("phy prof", load_from_json("results/mmlu_college_prof_phy_llama_30B_data_ntrain_0/results.json", subjs, subj_for_macro_avg=subj_for_macro_avg)),
    ],

    # Chat-GPT
    # [
    #     # ("default", load_from_json("results/tomi_default_new_gpt-3.5-turbo_data_tomi_50_mcq_shuf_ntrain_0/results.json", subjs, subj_for_macro_avg=subj_for_macro_avg)),
    #     ("3 yo", load_from_json("results/tomi_age_0_new_gpt-3.5-turbo_data_tomi_50_mcq_shuf_ntrain_0/results.json", subjs, subj_for_macro_avg=subj_for_macro_avg)),
    #     ("5 yo", load_from_json("results/tomi_age_1_new_gpt-3.5-turbo_data_tomi_50_mcq_shuf_ntrain_0/results.json", subjs, subj_for_macro_avg=subj_for_macro_avg)),
    #     ("10 yo", load_from_json("results/tomi_age_2_new_gpt-3.5-turbo_data_tomi_50_mcq_shuf_ntrain_0/results.json", subjs, subj_for_macro_avg=subj_for_macro_avg)),
    #     ("15 yo", load_from_json("results/tomi_age_3_new_gpt-3.5-turbo_data_tomi_50_mcq_shuf_ntrain_0/results.json", subjs, subj_for_macro_avg=subj_for_macro_avg)),
    #     ("30 yo", load_from_json("results/tomi_age_4_new_gpt-3.5-turbo_data_tomi_50_mcq_shuf_ntrain_0/results.json", subjs, subj_for_macro_avg=subj_for_macro_avg)),
    # ],
]

test_names = data[0]
baselines = [lab for lab, _ in data[1]]

# Extract the data
labels = data[0]
models = data[1]

# Create the plot
fig, ax = plt.subplots(figsize=(14, 6))
ax.set_ylim(bottom=0.0, top=1.05)
for model_name, model_data in models:
    x = range(len(labels))
    y = model_data  # remove the "Avg (micro)" score
    ax.plot(x, y, label=model_name, marker='o', linewidth=3)

# Set the axes labels and title
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels)
ax.set_xlabel("Subjects")
ax.set_ylabel("Accuracy")
ax.set_title(title)

# Add a legend
ax.legend()

# Display the plot
plt.show()
