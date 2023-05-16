import os
import json
import matplotlib.pyplot as plt
import re
import numpy as np
from termcolor import colored

all_values_ = []
def extract_value(directory, key="_lotr_character_"):
    label = os.path.basename(directory)
    if key in label:
        start_index = label.find(key) + len(key)

    elif "_ntrain_" in label:
        start_index = label.find("ntrain_") + len("ntrain_") + 1

    else:
        start_index = 0

    if "__2023" in label:
        end_index = label.find("__2023")
    elif "_2023" in label:
        end_index = label.find("_2023")
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


def plot_baseline(data, ax, directory, offset, keys_to_plot=None, subj=None, bar_width=1.0, min_bar_size=0.1, horizontal_bar=False, value_limit=250):

    if subj:
        draw_metrics = data['metrics'][subj]

    else:
        # only one subject
        subjects = list(data['metrics'].keys())

        if len(subjects) == 1:
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

        # add averages
        mean_college_perf = subjects_average(subjects_to_average=[
            "college_biology",
            "college_chemistry",
            "college_computer_science",
            "college_mathematics",
            "college_medicine",
            "college_physics",
        ], metric='accuracy', data=data)

        if mean_college_perf is not None:
            draw_metrics["Mean"] = mean_college_perf

        mean_hs_perf = subjects_average(subjects_to_average=[
            "high_school_biology",
            "high_school_chemistry",
            "high_school_computer_science",
            "high_school_european_history",
            "high_school_geography",
            "high_school_mathematics",
            "high_school_physics",
            "high_school_psychology",
            "high_school_us_history",
            "high_school_world_history",
        ], metric='accuracy', data=data)

        if mean_hs_perf is not None:
            draw_metrics["Mean"] = mean_hs_perf

        mean_tom = subjects_average(subjects_to_average=[
            "tomi_second_order_tom",
            "tomi_first_order_tom",
            "tomi_second_order_no_tom",
            "tomi_first_order_no_tom"
        ], metric='accuracy', data=data)

        if mean_tom is not None:
            draw_metrics["Mean ToM"] = mean_tom

        mean_tom_fo = subjects_average(subjects_to_average=[
            "tomi_first_order_tom",
            "tomi_first_order_no_tom"
        ], metric='accuracy', data=data)

        if mean_tom_fo is not None:
            draw_metrics["Mean ToM (fo)"] = mean_tom_fo

    if keys_to_plot is None:
        # use all the keys
        keys = list(draw_metrics.keys())
    else:
        keys = keys_to_plot

    key_indices = {key: i for i, key in enumerate(keys)}

    values = [draw_metrics[key] for key in keys]

    key = None
    # key = "Favorite music genre"
    # key = "Political organization of the country of growing up"
    # key = "Political system of home country"
    # key = "Type of home society"
    # key = "Favorite movie character"
    # key = "hobbies"

    # label = extract_by_key(directory, key=key)
    if "profile" in directory:
        label = extract_profile(directory)

    elif "lotr_character" in directory:
        label = extract_value(directory, "_lotr_character_")
    elif "music_expert" in directory:
        label = extract_value(directory, "_music_expert_")
    elif "music_AI_experts" in directory:
        label = extract_value(directory, "_music_expert_")
    elif "hobby" in directory:
        label = extract_value(directory, "_hobby_")
    else:
        label = os.path.basename(directory)

    label = label.rstrip("_").lstrip("_")

    x_values = [key_indices[key] + offset for key in keys]

    x_values = [v+bar_width/2 for v in x_values]


    def color_for_edge(label):
        label_to_color = {
            # "Rome": "darksalmon",
            # "Spain": "black",
        }

        for k, v in label_to_color.items():
            if k in label:
                return v

        return None

    def color_for_label(label):
        label_to_color = {
            "Democracy": "blue",
            "Theocracy": "black",
            # "Communism": "red",
            "Totalitarianism": "gold",
            "Anarchy": "brown",

            # "Christianity": "lightyellow",
            # "Pagan": "gray"
        }

        for k, v in label_to_color.items():
            if k in label:
                return v

        return None

    # create dummy bars if 0
    v_to_add = []
    x_to_add = []

    all_values_.append(values)

    for ind, v in enumerate(values):
        if abs(v) < bar_width/2:
            v_to_add.extend([min_bar_size, -min_bar_size])
            x_to_add.extend([x_values[ind], x_values[ind]])

    values = values + v_to_add
    x_values = x_values + x_to_add

    if horizontal_bar:
        # draw a horizontal bar
        ax.barh(x_values, values, label=label, height=bar_width, color=color_for_label(label))

    else:
        if figure_draw:
            labels = [label] * len(values)

            bar_color_dict = {
                'Conformity': "orange",
                'Tradition': "orange",
                'Benevolence': "green",
                'Universalism': "green",
                'Self-Direction': "blue",
                'Stimulation': "blue",
                'Hedonism': "blue",
                'Achievement': "red",
                'Power': "red",
                'Security': "orange"
            }

            for i, (value, label, key) in enumerate(zip(values, labels, keys)):
                print(key)
                plt.bar(x_values[i], value, label=label, width=bar_width, color=bar_color_dict[key])

        else:
            ax.bar(x_values, values, label=label, width=bar_width, color=color_for_label(label))
            #, facecolor=color_for_label(label), edgecolor=color_for_edge(label), linewidth=2)

    if args.horizontal_bar:
        assert all([-value_limit <= v <= value_limit for v in values])
        ax.set_xlim([-value_limit, value_limit])

        ax.set_ylabel('Values', fontsize=15)
        ax.set_xlabel('Scores', fontsize=15)

    else:
        # set y-axis limits
        if test_set_name == "pvq_male":
            ax.set_ylim([0, 6.1])

        elif test_set_name == "hofstede":
            ax.set_ylim([-350, 350])

        elif test_set_name == "big5_50":
            ax.set_ylim([0, 55])

        elif test_set_name == "big5_100":
            ax.set_ylim([0, 110])

        else:
            ax.set_ylim([0, max([6, *values])+0.1])

        ax.set_xlabel('Values', fontsize=15)
        ax.set_ylabel('Scores', fontsize=15)

    # if "gpt-3.5-turbo-0301" in directory:
    #     ax.set_title("gpt-3.5-turbo-0301")
    #
    # elif "gpt-4-0314" in directory:
    #     ax.set_title("gpt-4-0314")

    if not args.separate_legend:
        ax.legend(loc="best", fontsize=15)

    return keys

figure_draw = False

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('directories', nargs='+', help='directories containing results.json files')
    parser.add_argument('--save', action="store_true")
    parser.add_argument('--separate_legend', action="store_true")
    parser.add_argument('--filename', type=str, default="hobbies_pvq")
    parser.add_argument('--horizontal-bar', '-hb', action="store_true")
    args = parser.parse_args()

    keys_to_plot = None
    # keys_to_plot = ['Conformity', 'Tradition', 'Benevolence', 'Universalism', 'Self-Direction', 'Stimulation', 'Hedonism', 'Achievement', 'Power', 'Security']
    # keys_to_plot = ["power_distance", "individualism", "masculinity", "uncertainty_avoidance", "long_term_orientation", "indulgence"]

    bar_width = 0.10
    bar_margin = 1.2

    if figure_draw:
        bar_width = 0.9

    mean_primary_value_alignment = None

    fig, ax = plt.subplots(figsize=(12, 6))

    all_bars_width = len(args.directories) * (bar_width*bar_margin)  # bars with margins

    keys = None

    # chronological order
    directories = args.directories

    for soc_type in ["Hunter-Gatherer", "Horticultural and Pastoral", "Agricultural", "Industrial", "Postindustrial"]:
        for d in directories:
            if soc_type in d:
                directories.append(d)

    if any(["Age" in d for d in directories]):
        # extract the number after Age:
        def sort_by_key(directories, key):
            dir_2_value = {}
            for dir in directories:
                value = extract_by_key(dir, key)
                value = int(value) if value.isdigit() else -1
                dir_2_value[dir] = value

            # sort by value
            sorted_directories = sorted(directories, key=dir_2_value.get)
            return sorted_directories

        directories = sort_by_key(directories, "Age")

    # remove directories which contain substrings from the list
    ignore_patterns = []
    ignore_patterns = ["gen_space", "gen_w_space"]
    print("Ignoring patterns: ", ignore_patterns)

    for substring in ignore_patterns:
        directories = [d for d in directories if substring not in d]

    directories = [d for d in directories if os.path.isfile(os.path.join(d, 'results.json'))]

    if "pvq_test" in directories[0] or "pvq" in directories[0]:
        test_set_name = "pvq_male"
    elif "hofstede" in directories[0]:
        test_set_name = "hofstede"
    elif "big5_50" in directories[0]:
        test_set_name = "big5_50"
    elif "big5_100" in directories[0]:
        test_set_name = "big5_100"
    else:
        test_set_name = "pvq_male"

    current_max_y = 0

    dir_2_data = {}

    for i, directory in enumerate(directories):
        if not os.path.isdir(directory):
            continue

        results_json_path = os.path.join(directory, 'results.json')
        if not os.path.isfile(results_json_path):
            continue

        with open(results_json_path, 'r') as f:
            data = json.load(f)

        offset = -all_bars_width/2 + (i/len(args.directories))*all_bars_width
        keys_ = plot_baseline(data, ax, directory, offset, keys_to_plot=keys_to_plot, bar_width=bar_width, min_bar_size=0.05, horizontal_bar=args.horizontal_bar)

        dir_2_data[directory] = data

        # check that keys are the same in all the baselines
        assert keys is None or keys_ == keys
        keys = keys_

    # variance over baselines per value
    variances_ = np.stack([list(d["metrics"][test_set_name].values()) for d in dir_2_data.values()]).var(axis=0)  # todo: remove variances_

    variances = np.array(all_values_).var(axis=0)
    assert all(variances_ == variances)

    if test_set_name == "pvq_male":
        test_set_values = [
            'Conformity',
            'Tradition',
            'Benevolence',
            'Universalism',
            'Self-Direction',
            'Stimulation',
            'Hedonism',
            'Achievement',
            'Power',
            'Security'
        ]
    elif test_set_name == "hofstede":
        test_set_values = [
            "Power Distance",
            "Masculinity",
            "Uncertainty Avoidance",
            "Long-Term Orientation",
            "Indulgence",
            "Individualism"
        ]
    elif test_set_name in ["big5_50", "big5_100"]:
        test_set_values = [
            "Neuroticism",
            "Extraversion",
            "Openness to Experience",
            "Agreeableness",
            "Conscientiousness"
        ]

    primary_value_alignments = []
    if all(["Primary Values".lower() in d.lower() for d in directories]):
        for dir, data in dir_2_data.items():


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

            # map_prim_values = {
            #     "long term orientation": "long_term_orientation",
            #     "power distance": 'power_distance',
            #     "uncertainty avoidance": "uncertainty_avoidance",
            # }
            # primary_values = [map_prim_values.get(p, p) for p in primary_values]

            if "Primary values" not in profile:
                raise ValueError(f"Primary values are not in the profile: {profile}.")

            assert all([prim_v in test_set_values for prim_v in primary_values])

            # compute the metrics avg_{prim_values} - avg_{other_values}
            # avg_primary_values = np.mean([data['metrics'][test_set_name][val] for val in primary_values])
            # avg_other_values = np.mean([data['metrics'][test_set_name][val] for val in list(set(test_set_values) - set(primary_values))])
            # primary_value_alignment = avg_primary_values - avg_other_values
            from collections import defaultdict
            normalizing_constants = {
                    "pvq_male": defaultdict(lambda: 5),
                    "hofstede": {
                        "Power Distance": 2*300,
                        "Individualism": 2*350,
                        "Masculinity": 2*350,
                        "Uncertainty Avoidance": 2*325,
                        "Long-Term Orientation": 2*325,
                        "Indulgence": 2*375,
                    },
                    "big5_50": defaultdict(lambda: 50),
            }

            normalizing_offset = {
                "pvq_male": defaultdict(lambda: -1),
                "hofstede": {
                    "Power Distance": 0,
                    "Individualism": 0,
                    "Masculinity": 0,
                    "Uncertainty Avoidance": 0,
                    "Long-Term Orientation": 0,
                    "Indulgence": 0,
                },
                "big5_50": defaultdict(lambda: 0),
            }

            # print("No normalization")
            # normalizing_constants = defaultdict(lambda :defaultdict(lambda:1))
            # normalizing_offset = defaultdict(lambda :defaultdict(lambda:0))

            for perm_metrics in data["per_permutation_metrics"]:

                # we normalize by (x+normalizing_offset)/normalizing_constant

                norm_perm_metrics = {
                    val: (
                                 perm_metrics[test_set_name][val]+normalizing_offset[test_set_name][val]
                    )/normalizing_constants[test_set_name][val] for val in test_set_values
                }

                avg_primary_values = np.mean([norm_perm_metrics[val] for val in primary_values])
                avg_other_values = np.mean(
                    [norm_perm_metrics[val] for val in list(set(test_set_values) - set(primary_values))])
                perm_alignment = avg_primary_values - avg_other_values
                # print("permutation alignment: ", perm_alignment)
                primary_value_alignments.append(perm_alignment)

            perspective_value_alignment = np.mean(primary_value_alignments[-len(data["per_permutation_metrics"]):])
            # print(f"Primary value alignment for {primary_values}: {perspective_value_alignment}.")

        # todo: confirm this
        mean_primary_value_alignment = np.mean(primary_value_alignments)
        print(colored(f"Mean primary value alignment (over all): {round(mean_primary_value_alignment, 3)}", "green"))

    # mean over value dimensions
    mean_variance = variances.mean()

    # print(f"Mean (over values) Variance (over baselines): {mean_variance}")

    if args.horizontal_bar:

        # Set the y-ticks labels on the left side
        y_locs = list(range(len(keys)))

        # right labels
        key_to_hofstede_label_right = {
            "power_distance": "high power distance",
            "individualism": "individualistic",
            "masculinity": "masculine",
            "uncertainty_avoidance": "high uncertainty_avoidance",
            "long_term_orientation": "long term orientation",
            "indulgence": "indulgence",
        }

        key_to_hofstede_label_left = {
            "power_distance": "low power distance",
            "individualism": "collectivistic",
            "masculinity": "feminine",
            "uncertainty_avoidance": "low uncertainty avoidance",
            "long_term_orientation": "short term orientation",
            "indulgence": "restraint",
        }

        left_labels = [key_to_hofstede_label_left.get(k) for k in keys]
        right_labels = [key_to_hofstede_label_right.get(k) for k in keys]

        # add left ticks
        ax.set_yticks(y_locs)
        ax.set_yticklabels(left_labels)

        ax2 = ax.twinx()
        ax2.set_ylim(ax.get_ylim())
        ax2.set_yticks(y_locs)
        ax2.set_yticklabels(right_labels)

    else:
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels(keys, rotation=45)

    if figure_draw:
        ax.legend().remove()
        ax.set_title("")

    if args.save:
        for ext in ["png", "svg"]:
            savepath = f"visualizations/{args.filename}.{ext}"
            print(f"Saved to: {savepath}")
            plt.tight_layout()
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

