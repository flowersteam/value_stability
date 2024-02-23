import argparse
from collections import Counter
import json
import sys
import os
import numpy as np
from itertools import chain


def process_json(json_path, show_values=False):
    json_data = load_json_file(json_path)

    json_data['answers']
    answers = json_data['answers']

    test_set = "pvq_male"
    # test_set = "tolkien_ultimatum"
    # if the biggest logprob is -np.inf -> the answer was refused
    pc_refused = np.stack(
        [np.array(list(lps.values())[0]).max(axis=1) == -np.inf for lps in json_data['lprobs']]
    ).mean()
    print("% refused:", pc_refused)

    n_part_colapsed = np.sum([1 == len(set(np.array(a['pvq_auto'])[:, 1])) for a in answers])
    print("n part collapsed", n_part_colapsed)
    print("% part collapsed", n_part_colapsed/len(answers))

    all_ans = []
    for i in range(len(answers)):
        all_ans.append([x for _, x in list(answers[i].values())[0]])

    pc_most_commons = []
    for part_ans in all_ans:
        most_common = find_most_common_element(part_ans)
        pc_most_commons.append(np.mean([a == most_common for a in part_ans]))
    print("Most common per part:", np.mean(pc_most_commons))

    most_common = find_most_common_element(chain(*all_ans))
    pc_most_common = np.mean([a == most_common for a in chain(*all_ans)])

    print(f"Total: % most common({most_common}): {pc_most_common}")

    acceptances = np.array(all_ans) == 1

    # splits = np.split(acceptances, 8, 1)

    if show_values:
        for metr, part in zip(json_data['per_simulated_participant_metrics'], json_data["simulated_population"]):

            if part in ["Morgoth", "Sauron", "Gandalf", "Frodo Baggins", "Peregrin Took"]:
                print("---------------")
                print(part)
                print(metr)

                # metr_names = list(metr[test_set].keys())[1:]
                # accs_fair = np.array(accs)[:, :5]
                # accs_unfair = np.array(accs)[:, 6:]
                # # we left 50-50 out

                # for metr_name, a_f, a_uf in zip(metr_names, accs_fair, accs_unfair):
                #     print(f'{metr_name.replace("Acceptance Rate ", "")} : fair - {np.mean(a_f)}  unfair - {np.mean(a_uf)})')

                # print(" ".join([f"{val.replace('Acceptance Rate ','')}:{sc:.2f}" for val, sc in metr[test_set].items()]))


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
    "tax",
    "vacation",
    "religion",
]
# Function to process directories and load JSON files
def process_directories(dirs, show_values=False):

    to_process = []
    for dir in dirs:
        if any([p in dir for p in ignore_patterns]):
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

    for file_path in to_process:
        print("File:", file_path)
        process_json(file_path, show_values)


# Main function
def main():
    parser = argparse.ArgumentParser(description='Process some directories.')
    parser.add_argument('-v', '--show-values', action='store_true', help='Flag description')
    parser.add_argument('dirs', nargs='*', help='Directories to process')

    # Parse the arguments
    args = parser.parse_args()

    process_directories(args.dirs, args.show_values)

if __name__ == "__main__":
    main()
