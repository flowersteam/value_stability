import argparse
from collections import Counter
import json
import sys
import os
import numpy as np
from itertools import chain
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib.lines import Line2D

# GPT-4 classification
good_guys = [
    "Gandalf", "Aragorn", "Celeborn", "Galadriel", "Tom Bombadil", "Elrond", "Frodo Baggins", 
    "Finrod Felagund", "Glorfindel", "Goldberry", "Bilbo Baggins", "Faramir", "Éowyn", 
    "Samwise Gamgee", "Théoden", "Thranduil", "Beorn", "Arwen", "Halbarad", "Celebrimbor", 
    "Gil-galad", "Meriadoc Brandybuck", "Treebeard", "Radagast", "Elendil", "Éomer", "Legolas", 
    "Thorin Oakenshield", "Peregrin Took", "Eärendil", "Elwing", "Lúthien", "Beren", "Tuor", 
    "Idril", "Finwë", "Míriel", "Melian", "Gimli"
]

neutral_guys = [
    "Sméagol", "Maedhros", "Fëanor", "Boromir", "Túrin Turambar", "Isildur", "Denethor", "Húrin", "Thingol"
]

bad_guys = [
    "Gollum", "Sauron", "Saruman", "Smaug", "Morgoth", "Gothmog (Balrog)", "Lungorthin", 
    "Shelob", "Gríma Wormtongue", "Ungoliant", "Thuringwethil", "Durin's Bane"
]

values = [
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

def process_json(json_path):
    json_data = load_json_file(json_path)

    average_per_part_answer = [
        np.array(answers['pvq_auto'])[:, 1].astype(float).mean() for answers in json_data["answers"]
    ]

    return {
        name: [
            metrics['pvq_auto'][v] - avg_part_answer for v in values
        ] for name, metrics, avg_part_answer in zip(
            json_data['simulated_population'],
            json_data['per_simulated_participant_metrics'],
            average_per_part_answer
        )
    }


def load_json_file(file_path):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            return data
    except FileNotFoundError:
        print("File not found.")
    except json.JSONDecodeError:
        print("Error decoding JSON.")
    except Exception as e:
        print(f"An error occurred: {e}")

        
def find_most_common_element(lst):
    if not lst:
        return None

    count = Counter(lst)
    most_common = count.most_common(1)[0][0]
    return most_common


# Function to load a JSON file
def load_json_file(file_path):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except json.JSONDecodeError:
        print(f"Error decoding JSON in file: {file_path}")
    except Exception as e:
        print(f"An error occurred while processing {file_path}: {e}")


ignore_patterns = [
    # "tax",
    # "vacation",
    # "religion",
    "format_chat___",
]

# Function to process directories and load JSON files
def process_directories(dirs):

    to_process = []
    for dir in dirs:
        if any([pat in dir for pat in ignore_patterns]):
            continue

        if os.path.isdir(dir):
            for filename in os.listdir(dir):
                if filename.endswith('.json'):
                    file_path = os.path.join(dir, filename)
                    to_process.append(file_path)


        elif os.path.isfile(dir) and dir.endswith('.json'):
            to_process.append(dir)

        else:
            print(f"{dir} is not a valid directory or JSON file")

    participant_values_ = []
    for file_path in to_process:
        participant_values_.append(process_json(file_path))

    names = list(participant_values_[0].keys())

    # merge contexts
    participant_values = {}
    for name in names:
        participant_values[name] = np.array(list(chain(*[part_v[name] for part_v in participant_values_])))

    return participant_values


# Main function
def main():
    parser = argparse.ArgumentParser(description='Process some directories.')
    parser.add_argument('dirs', nargs='*', help='Directories to process')

    # Parse the arguments
    args = parser.parse_args()
    participant_values = process_directories(args.dirs)

    names = list(participant_values.keys())

    data = np.stack([participant_values[n] for n in names])

    values_names = ["Power", "Achievement", "Benevolence", "Universalism", "Tradition", "Conformity", "Security", "Self-Direction", "Stimulation", "Hedonism"]

    # min_ = np.min(data)
    # max_ = np.max(data)
    # mean_ = np.mean(data)
    # var_ = np.var(data)
    #
    # vals = []
    # # todo: make pairs correct -> sometimes 3 someteimes 2 etc
    values_names_pairs = list(zip(values_names[::2], values_names[1::2]))

    # values_names_pairs = [["Power", "Achievement"], ["Benevolence", "Universalism"]]
    # names = names+values_names_pairs
    # for val_names in values_names_pairs:
    #     vals.append(np.array([4 if v in val_names else -1 for v in values] * 5))

    # names = names+values_names
    # for val_name in values_names:
    #     # 6 1
    #     vals.append(np.array([4.5 if v == val_name else -0.5 for v in values] * 5))
    #     # 6 3.5
    #     # vals.append(np.array([2.25 if v == val_name else -0.25 for v in values] * 5))

    # data = np.vstack([data, np.array(vals)])

    # powe, ach, ben, uni
    # data = np.hstack([data[:,i::10] for i in range(4)])

    pca = PCA(n_components=2)
    print(f"Data shape: {data.shape}")
    data = StandardScaler().fit_transform(data)
    data_trans = pca.fit_transform(data)

    # pca_trans = data_trans[:-10]
    # vals_trans = data_trans[-10:]
    pca_trans = data_trans

    R2_1, R2_2 = pca.explained_variance_ratio_
    pca_1 = pca_trans[:, 0]
    pca_2 = pca_trans[:, 1]

    plt.figure(figsize=(10, 10))
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.xlabel(r'PC 1 ($R^{2}=' + f'{R2_1:.2f}' + '$)', fontsize=25)
    plt.ylabel(r'PC 2 ($R^{2}=' + f'{R2_2:.2f}' + '$)', fontsize=25)

    text_fontsize=10

    colors = []

    for name in names:
        if name in bad_guys:
            color = "red"
        elif name in good_guys:
            color = "green"
        elif name in values_names or name in values_names_pairs:
            color = "blue"
        else:
            color = "black"

        colors.append(color)

    plt.scatter(pca_1, pca_2, c=colors, s=50)

    for x, y, n in zip(pca_1, pca_2, names):
        plt.text(x+0.15, y+0.15, n, fontsize=text_fontsize)

    # for val_name, val_trans in zip(values_names, vals_trans):
    #     plt.scatter([val_trans[0]], [val_trans[1]], marker="x")
    #     plt.text(val_trans[0], val_trans[1], val_name)

    # plt.scatter([power_pca[0]], [power_pca[1]], c="red", marker="x")
    # plt.scatter([ben_pca[0]], [ben_pca[1]], c="green", marker="x")
    # plt.scatter([tr_pca[0]], [tr_pca[1]], c="yellow", marker="x")
    # plt.scatter([hed_pca[0]], [hed_pca[1]], c="blue", marker="x")

    # plt.text(power_pca[0], power_pca[1], "power")
    # plt.text(ben_pca[0], ben_pca[1], "benevolence")
    # plt.text(tr_pca[0], tr_pca[1], "tradition")
    # plt.text(hed_pca[0], hed_pca[1], "hedonism")

    # plt.legend(targets, prop={'size': 15})
    # plt.show()

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Positive ch.', markerfacecolor='green', markersize=20),
        Line2D([0], [0], marker='o', color='w', label='Neutral ch.', markerfacecolor='black', markersize=20),
        Line2D([0], [0], marker='o', color='w', label='Negative ch.', markerfacecolor='red', markersize=20),
    ]

    # Add the legend to the plot
    plt.legend(handles=legend_elements, fontsize=23)

    savepath = f"visualizations/pca.svg"
    plt.savefig(savepath)
    print(f"Saved to: {savepath}")

    plt.show()


if __name__ == "__main__":
    main()
