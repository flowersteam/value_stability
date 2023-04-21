import matplotlib.pyplot as plt
import json

transpose = False


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

subj_for_macro_avg = [
    "tomi_first_order_no_tom",
    "tomi_first_order_tom",
    "tomi_memory",
    "tomi_reality",
    "tomi_second_order_no_tom",  # no tom
    "tomi_second_order_tom",  # tom
]

subjs = ["tomi_first_order_no_tom", "tomi_first_order_tom", "tomi_memory", "tomi_reality", "tomi_second_order_no_tom", "tomi_second_order_tom", "average"]


title = "gpt-3.5"
title = ""
data = [
    # x-axis
    [subj_to_label[s] for s in subjs] + ["Avg (macro)"],
    [
        ("default", load_from_json("results/results_tomi_age/tomi_test_gpt-3.5-turbo-0301_data_tomi__2023_04_18_10_58_34/results.json", subjs, subj_for_macro_avg=subj_for_macro_avg)),
        ("3 yo", load_from_json("results/results_tomi_age/tomi_test_gpt-3.5-turbo-0301_data_tomi_ntrain_0_profile_Age:3_2023_04_18_10_40_13/results.json", subjs, subj_for_macro_avg=subj_for_macro_avg)),
        ("5 yo", load_from_json("results/results_tomi_age/tomi_test_gpt-3.5-turbo-0301_data_tomi_ntrain_0_profile_Age:5_2023_04_18_11_07_41/results.json", subjs, subj_for_macro_avg=subj_for_macro_avg)),
        ("10 yo", load_from_json( "results/results_tomi_age/tomi_test_gpt-3.5-turbo-0301_data_tomi_ntrain_0_profile_Age:10_2023_04_18_11_09_52/results.json", subjs, subj_for_macro_avg=subj_for_macro_avg)),
        ("30 yo", load_from_json("results/results_tomi_age/tomi_test_gpt-3.5-turbo-0301_data_tomi_ntrain_0_profile_Age:30_2023_04_18_11_12_02/results.json", subjs, subj_for_macro_avg=subj_for_macro_avg)),
    ]
]

test_names = data[0]
baselines = [lab for lab, _ in data[1]]

# transpose data
data_ = [
    # x-axis
    baselines,
    [
        (test_name, []) for test_name in test_names
    ]
]
for b_i in range(len(baselines)):
    for test_i in range(len(test_names)):
        data_[1][test_i][1].append(
            data[1][b_i][1][test_i]
        )

# replace data with transposed data
if transpose:
    data = data_
    print("Drawing transposed data (x-axis - baselines).")


# invert data

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
