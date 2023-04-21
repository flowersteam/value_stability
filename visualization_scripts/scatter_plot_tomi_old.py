import matplotlib.pyplot as plt
import json

transpose = False

# # k=3
# data = [
#     [
#         "First_order(TB)",  # no tom
#         "First_order(FB)",  # tom
#         "Memory",
#         "Reality",
#         "Second_order(TB)",  # no tom
#         "Second_order(FB)",  # tom
#         "Avg (micro)"
#     ],
#     [
#         ("ChatGPT"  , [0.973, 0.909, 1.000, 1.000, 0.917, 0.542, 0.875]),
#         ("LLaMa 7B" , [0.811, 1.000, 1.000, 0.833, 0.917, 0.917, 0.892]),
#         ("LLaMa 13B", [0.973, 1.000, 1.000, 1.000, 0.833, 0.792, 0.917]),
#         ("LLaMa 30B", [0.946, 1.000, 1.000, 0.958, 0.875, 0.917, 0.938]),
#     ]
# ]

# k=1 (in-context 50 years old)


def load_from_json(path, subjs, subj_for_macro_avg=None):
    with open(path, "r") as read_file:
        accs_json = json.load(read_file)

    subject_accs = [accs_json[s] for s in subjs]

    if subj_for_macro_avg is not None:
        macro_avg = sum([accs_json[s] for s in subj_for_macro_avg])/len(subj_for_macro_avg)
        subject_accs += [macro_avg]

    return subject_accs


# subj_to_label = {
#     "tomi_first_order_no_tom": "First_order TB (74)",  # no tom
#     "tomi_first_order_tom": "First_order FB (22)",  # tom
#     "tomi_memory": "Memory (48)",
#     "tomi_reality": "Reality (48)",
#     "tomi_second_order_no_tom": "Second_order(TB) (70)",  # no tom
#     "tomi_second_order_tom": "Second_order(FB) (26)",  # tom
#     "average": "Avg (micro)"
# }
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
    # torchrun --nproc_per_node 4 evaluate.py -k 0 -d data_tomi_50_mcq_shuf -e llama_30B -n tomi_default_new
    # torchrun --nproc_per_node 4 evaluate.py -k 0 -d data_tomi_50_mcq_shuf -e llama_30B -n tomi_age_<>_new
    #     ("3 yo", load_from_json("results/tomi_age_0_new_llama_30B_data_tomi_50_mcq_shuf_ntrain_0/results.json", subjs, subj_for_macro_avg=subj_for_macro_avg)),

    # # LLama
    [
        # ("random", load_from_json("results/tomi_default_new_dummy_data_tomi_50_mcq_shuf_ntrain_0/results.json", subjs, subj_for_macro_avg=subj_for_macro_avg)),
        # ("default", load_from_json("results/tomi_default_new_llama_30B_data_tomi_50_mcq_shuf_ntrain_0/results.json", subjs, subj_for_macro_avg=subj_for_macro_avg)),
        ("3 yo", load_from_json("results/tomi_age_0_new_llama_30B_data_tomi_50_mcq_shuf_ntrain_0/results.json", subjs, subj_for_macro_avg=subj_for_macro_avg)),
        ("5 yo", load_from_json("results/tomi_age_1_new_llama_30B_data_tomi_50_mcq_shuf_ntrain_0/results.json", subjs, subj_for_macro_avg=subj_for_macro_avg)),
        ("10 yo", load_from_json("results/tomi_age_2_new_llama_30B_data_tomi_50_mcq_shuf_ntrain_0/results.json", subjs, subj_for_macro_avg=subj_for_macro_avg)),
        ("15 yo", load_from_json("results/tomi_age_3_new_llama_30B_data_tomi_50_mcq_shuf_ntrain_0/results.json", subjs, subj_for_macro_avg=subj_for_macro_avg)),
        ("30 yo", load_from_json("results/tomi_age_4_new_llama_30B_data_tomi_50_mcq_shuf_ntrain_0/results.json", subjs, subj_for_macro_avg=subj_for_macro_avg)),
        ("50 yo", load_from_json("results/tomi_age_5_new_llama_30B_data_tomi_50_mcq_shuf_ntrain_0/results.json", subjs, subj_for_macro_avg=subj_for_macro_avg)),
    ],
    #
    # # Chat-GPT
    # [
    #     # ("default", load_from_json("results/tomi_default_new_gpt-3.5-turbo_data_tomi_50_mcq_shuf_ntrain_0/results.json", subjs, subj_for_macro_avg=subj_for_macro_avg)),
    #     ("3 yo", load_from_json("results/tomi_age_0_new_gpt-3.5-turbo_data_tomi_50_mcq_shuf_ntrain_0/results.json", subjs, subj_for_macro_avg=subj_for_macro_avg)),
    #     ("5 yo", load_from_json("results/tomi_age_1_new_gpt-3.5-turbo_data_tomi_50_mcq_shuf_ntrain_0/results.json", subjs, subj_for_macro_avg=subj_for_macro_avg)),
    #     ("10 yo", load_from_json("results/tomi_age_2_new_gpt-3.5-turbo_data_tomi_50_mcq_shuf_ntrain_0/results.json", subjs, subj_for_macro_avg=subj_for_macro_avg)),
    #     ("15 yo", load_from_json("results/tomi_age_3_new_gpt-3.5-turbo_data_tomi_50_mcq_shuf_ntrain_0/results.json", subjs, subj_for_macro_avg=subj_for_macro_avg)),
    #     ("30 yo", load_from_json("results/tomi_age_4_new_gpt-3.5-turbo_data_tomi_50_mcq_shuf_ntrain_0/results.json", subjs, subj_for_macro_avg=subj_for_macro_avg)),
    # ],
]
# # MCQ (k=0) 1000 test
# [
#     ("3 years old" , []),
#     # ("5 years old" , []),
#     # ("10 years old", []),
#     # ("15 years old", []),
#     ("30 years old", []),
# ],
# # MCQ (k=0) 250 test
# [
#     ("3 years old" , [0.020, 0.959, 1.000, 0.072, 0.006, 0.984, 0.373]),
#     # ("5 years old" , []),
#     # ("10 years old", []),
#     # ("15 years old", []),
#     ("30 years old", [0.289, 0.722, 1.000, 0.361, 0.068, 0.942, 0.485]),
# ],


########
########
##  the most important experiment
# MCQ (k=0) 50 test
# [
#     ("default (k=0)", [0.243, 0.727, 0.938, 0.104, 0.229, 0.615, 0.403]),
#     # ("default-no undef (k=0)", [0.838, 0.091, 0.458, 0.833, 0.829, 0.385, 0.674]),
#     ("3 years old (k=0)", [0.014, 1.000, 1.000, 0.042, 0.000, 1.000, 0.344]),
#     ("5 years old (k=0)", [0.108, 0.955, 1.000, 0.146, 0.029, 1.000, 0.389]),
#     ("10 years old (k=0)", [0.365, 0.818, 0.958, 0.479, 0.129, 0.923, 0.510]),
#     ("15 years old (k=0)", [0.432, 0.636, 0.979, 0.417, 0.100, 0.923, 0.500]),
#     ("30 years old (k=0)", [0.432, 0.636, 0.979, 0.438, 0.071, 0.962, 0.500]),
# ],

# # MCQ (k=0) 50 test - NO UNDEF options presented
# [
#     # ("default undef (k=0)", [0.243, 0.727, 0.938, 0.104, 0.229, 0.615, 0.403]),
#     ("default (k=0)", [0.838, 0.091, 0.458, 0.833, 0.829, 0.385, 0.674]),
#     ("3 years old (k=0)", [0.027, 1.000, 1.000, 0.312, 0.014, 1.000, 0.396]),
#     ("5 years old (k=0)", [0.176, 0.955, 1.000, 0.458, 0.043, 0.962, 0.458]),
#     ("10 years old (k=0)", [0.419, 0.682, 1.000, 0.646, 0.186, 0.846, 0.556]),
#     ("15 years old (k=0)", [0.514, 0.545, 1.000, 0.646, 0.243, 0.731, 0.573]),
#     ("30 years old (k=0)", [0.581, 0.409, 1.000, 0.750, 0.257, 0.846, 0.611]),
#     ("50 years old (k=0)", [0.432, 0.455, 1.000, 0.708, 0.214, 0.885, 0.562]),
#     ("dog (k=0)", [0.419, 0.364, 0.917, 0.896, 0.143, 0.769, 0.542]),
# ],

# MCQ (k=0) 50 test - "This is a questionnaire" - NO UNDEF options presented
# [
#     # ("default (k=0)", [0.743, 0.273, 0.958, 0.875, 0.500, 0.615, 0.694]),
#     ("3 years old (k=0)", [0.811, 0.227, 0.917, 0.750, 0.457, 0.538, 0.663]),
#     ("5 years old (k=0)", [0.865, 0.045, 0.833, 0.938, 0.671, 0.423, 0.722]),
#     ("10 years old (k=0)", [0.919, 0.045, 0.771, 0.958, 0.729, 0.385, 0.740]),
#     ("15 years old (k=0)", [0.892, 0.091, 0.750, 0.938, 0.700, 0.423, 0.726]),
#     ("30 years old (k=0)", [0.959, 0.045, 0.271, 1.000, 0.857, 0.077, 0.677]),
#     ("50 years old (k=0)", [0.946, 0.045, 0.333, 1.000, 0.886, 0.192, 0.701]),
# ],
########
########

# MCQ (k=5) 50 test
# [
#     ("default (k=5)", [1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000]),
#     ("3 years old (k=5)", [0.986, 1.000, 1.000, 1.000, 1.000, 1.000, 0.997]),
#     ("10 years old (k=5)", [1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000]),
#     ("30 years old (k=5)", [1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000]),
# ],

# # MCQ (k=1) 50 test
# [
#     ("default (k=1)" , [0.986, 0.045, 0.062, 0.979, 0.957, 0.000, 0.663]),
#     ("3 years old (k=1)" , [0.703, 0.682, 0.771, 0.771, 0.643, 0.269, 0.670]),
#     ("5 years old (k=1)" , [0.851, 0.455, 0.583, 0.875, 0.786, 0.077, 0.694]),
#     ("15 years old (k=1)", [0.986, 0.045, 0.146, 0.979, 0.943, 0.000, 0.674]),
#     ("30 years old (k=1)", [0.986, 0.045, 0.062, 1.000, 0.986, 0.000, 0.674]),
# ],


# GQA (k=1, 50yo)
# [
#     ("3 years old" , [0.919, 0.818, 1.000, 0.917, 0.875, 0.583, 0.861]),
#     ("5 years old" , [0.973, 0.909, 1.000, 0.917, 0.875, 0.583, 0.882]),
#     ("10 years old", [0.973, 0.909, 1.000, 0.917, 0.917, 0.542, 0.882]),
#     ("15 years old", [0.973, 0.909, 1.000, 0.917, 0.875, 0.542, 0.875]),
#     ("30 years old", [0.973, 0.909, 1.000, 0.917, 0.875, 0.542, 0.875]),
# ]
# GQA (k=3, 50yo)
# [
#     ("3 years old" , [0.973, 0.909, 1.000, 0.958, 0.917, 0.917, 0.951]),
#     ("10 years old", [1.000, 0.909, 1.000, 0.958, 0.958, 0.833, 0.951]),
#     ("30 years old", [1.000, 1.000, 1.000, 0.958, 0.917, 0.833, 0.951]),
# ]
# ]

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
from IPython import embed; embed()
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
