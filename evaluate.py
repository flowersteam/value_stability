import argparse
import random
import warnings
from collections import defaultdict
import hashlib
import datetime
import itertools
import string
import copy

from termcolor import colored, cprint

from utils import *
from svs_utils import *
from simulate_conversation_utils import *

from models import *

import numpy as np
import pandas as pd
import torch
import tiktoken

from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList, BitsAndBytesConfig
# from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training # for Qwen?

from personas.utils import simulated_participant_to_name


hf_cache_dir = get_hf_cache_dir()
os.environ['HF_HOME'] = hf_cache_dir


def create_permutation_dicts(args, n_options, choices, num_questions, population_size=None):


    # sample permutations based on given seed -> should correspond to different contexts
    original_state = random.getstate()  # save the original state
    random.seed(args.permute_options_seed)

    if len(set(n_options)) == 1:

        if n_options[0] > 9:
            raise ValueError("Number of options too big. Refactor code below to use it.")

        all_permutations = list(itertools.permutations(range(n_options[0])))

        permutations = random.choices(all_permutations, k=num_questions*population_size)
        permutations = [permutations[part_i:part_i+num_questions] for part_i in range(population_size)]

    else:
        # not all questions have the same number of options

        # string seed to int seed
        int_seed = int(hashlib.md5(args.permute_options_seed.encode('utf-8')).hexdigest(), 16)
        rng = np.random.default_rng(seed=int_seed)

        permutations = [
            [tuple(rng.permutation(n_options_q)) for n_options_q in n_options] for _ in range(population_size)
        ]

    permutations_dicts = [
        [
            dict(zip(choices, perm)) for perm in part_perms
        ] for part_perms in permutations
    ]

    # revert original state
    random.setstate(original_state)


    return permutations_dicts


def map_number_to_choice(number, inv_permutations_dict):
    choice = inv_permutations_dict[number-1]
    return choice


timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
print("timestamp:", timestamp)


hf_token = os.environ["HF_TOKEN"]
hidden_token = hf_token[:6] + "*" * (len(hf_token)-12) + hf_token[-6:]
print("HF TOKEN:", hidden_token)


choices = list(string.ascii_uppercase) + list(string.ascii_lowercase)

llama_dir = "/gpfswork/rech/imi/utu57ed/llama/llama_files/"


def get_prompt_skeleton(experiment_name, args, simulated_participant_description, base_model_template):

    if "pvq" in experiment_name:
        assert "pvq" in args.data_dir
        questionnaire_description = "Here we briefly describe some people. Please read each description and think about how much each person is or is not like you. Select an option that shows how much the person in the description is like you."

    elif "svs" in experiment_name:
        assert "svs" in args.data_dir
        with open('data/data_svs/raw/description.txt', 'r') as file:
            questionnaire_description = file.read().rstrip()

    elif "donation" in experiment_name:
        assert "donation" in args.data_dir
        questionnaire_description = ""

    elif "bag" in experiment_name:
        assert "bag" in args.data_dir
        questionnaire_description = ""

    elif "religion" in experiment_name:
        assert "religion" in args.data_dir
        questionnaire_description = ""

    else:
        raise ValueError(f"Experiment name is ill-defined {args.experiment_name}")

    if base_model_template:
        prefix = "The following is a conversation with"
    else:
        prefix = "You are"

    if args.simulated_population_config == "permutations":
        set_persona_str = ""
    else:
        set_persona_str = f"{prefix} {simulated_participant_description}"

    if args.query_prompt:
        query_str = args.query_prompt
    else:
        query_str = "Answer: ("

    prompt_skeleton = {
        "set_persona_str": set_persona_str,  # remove newline from the end
        "questionnaire_description": questionnaire_description,
        "query_str": f"{query_str}",
    }

    return prompt_skeleton


def format_example(df, idx, experiment_name, args, permutations_dict, simulated_participant_description, base_model_template=None):
    # an item contains a question and suggested answers
    item_str = df.iloc[idx, 0]
    k = df.shape[1] - 2

    # extract options
    num_options = 0
    options_strings = []
    for j in range(k):
        op_str = df.iloc[idx, j+1]

        if op_str == "undef":
            continue

        options_strings.append(op_str)

        num_options += 1

    item_str_ = item_str
    choices_prefixes = choices[:num_options]
    item_str_ += create_choices_str(choices_prefixes, [options_strings[permutations_dict[ch]] for ch in choices_prefixes])

    for ch in choices[:num_options]:
        item_str += "\n({}) {}".format(ch, options_strings[permutations_dict[ch]])

    # testing, remove the manual thing later (keep the function)
    assert item_str == item_str_


    prompt = get_prompt_skeleton(
        experiment_name=experiment_name,
        args=args,
        simulated_participant_description=simulated_participant_description,
        base_model_template=base_model_template
    )

    prompt["item_str"] = item_str

    return prompt, num_options



def eval(args, test_df, participant_perm_dicts, llm_generator=None, simulated_participant=None, opening_question=None, interlocutor="human"):
    cors = [None] * test_df.shape[0]
    all_probs = [None] * test_df.shape[0]
    all_lprobs = [None] * test_df.shape[0]
    all_answers = [None] * test_df.shape[0]
    all_generations = [None] * test_df.shape[0]
    all_scores = [None] * test_df.shape[0]

    # hashing for simulated conversations
    messages_conv = None

    gpt_token_counter = {"input": 0, "output": 0}

    assert test_df.shape[0] == len(participant_perm_dicts)

    for item_i, permutations_dict in enumerate(participant_perm_dicts):

        if item_i % 20 == 0:
            print(f"Eval progress: {item_i}/{test_df.shape[0]}")

        #  e.g. A -> A little like me
        label_2_text_option_dict = {
            label: test_df.iloc[item_i, score+1] for label, score in permutations_dict.items()
        }
        prompt, n_options = format_example(
            test_df, item_i,
            experiment_name=args.experiment_name,
            args=args,
            permutations_dict=permutations_dict,
            simulated_participant_description=simulated_participant["description"],
            base_model_template=llm_generator.base_model_template
        )

        assert n_options == len(permutations_dict)
        answers = choices[:n_options]

        assert all([a in permutations_dict for a in answers])

        label = test_df.iloc[item_i, test_df.shape[1]-1]
        assert label in answers + ["undef"]

        if args.estimate_gpt_tokens:
            gpt_tokenizer = tiktoken.get_encoding("cl100k_base")
        else:
            gpt_tokenizer = None

        if opening_question:

            set_persona_str = prompt["set_persona_str"]
            if messages_conv is None:
                assert item_i == 0
                if args.verbose:
                    print("SIMULATING CONVERSATION")

                messages_conv = simulate_conversation(
                    args=args,
                    opening_question=opening_question,
                    model_set_persona_string=set_persona_str,
                    simulated_participant=simulated_participant,
                    llm_generator=llm_generator,
                    interlocutor=interlocutor,
                )

                if args.estimate_gpt_tokens:
                    # topic setting msg
                    current_input_tokens = len(gpt_tokenizer.encode(messages_conv[0]['content']))

                    for msg_i in range(1, len(messages_conv)):
                        current_output_tokens = len(gpt_tokenizer.encode(messages_conv[msg_i]['content']))
                        gpt_token_counter['input'] += current_input_tokens
                        gpt_token_counter['output'] += current_output_tokens

                        # add for next message
                        current_input_tokens += current_output_tokens

            else:
                assert item_i != 0
                if args.verbose:
                    print("LOADING CACHED CONVERSATION")

        if args.estimate_gpt_tokens:
            # gpt params
            messages = construct_messages(
                prompt=prompt,
                system_message=True,
                messages_conv=messages_conv
            )
            n_input_tokens = sum([len(gpt_tokenizer.encode(msg['content'])) for msg in messages])

            gpt_token_counter['input'] += n_input_tokens
            gpt_token_counter['output'] += 1

        assert bool(messages_conv is None) == bool(opening_question is None)

        if "data_svs" in args.data_dir:
            assert "svs" in args.experiment_name

            # Select and rate the extreme values for both groups
            if item_i in svs_groups_start_indices:  # start indices are : 0, 30
                first_non_extreme_value = True

                group_id = first_index_to_svs_group[item_i]
                group_values = get_values_for_group(group_id)

                if group_id == 1:
                    assert group_values == list(test_df.iloc[:svs_group_to_size[group_id], 0])
                elif group_id == 2:
                    assert group_values == list(test_df.iloc[-svs_group_to_size[group_id]:, 0])

                # choose and rate the most and least important values (extreme values)
                group_values_to_choose_from = copy.copy(group_values)
                running_messages = messages_conv

                for extreme_value_str in ["most", "least"]:

                    # 1. Choose the extreme values (most or least important)
                    #################################################################
                    chosen_value, item_i_in_group, running_messages = choose_extreme_value(
                        group_id=group_id,
                        group_values=group_values,
                        extreme_value_str=extreme_value_str,
                        create_choices_str=create_choices_str,
                        choices=choices,
                        group_values_to_choose_from=group_values_to_choose_from,
                        prompt=prompt,
                        construct_messages=construct_messages,
                        llm_generator=llm_generator,
                        previous_messages=running_messages,
                        label_2_text_option_dict=label_2_text_option_dict,
                        simulated_participant=simulated_participant,
                    )

                    chosen_item_i = item_i_in_group + item_i
                    assert chosen_item_i == list(test_df.iloc[:, 0]).index(chosen_value)

                    if extreme_value_str == "most":
                        # when choosing the least important value, the most important value should among the options
                        group_values_to_choose_from.remove(chosen_value)

                    # 2. Score the extreme values
                    cor, lprobs, probs, pred, generation, score, running_messages = score_extreme_value(
                        format_example=format_example,
                        test_df=test_df,
                        chosen_item_i=chosen_item_i,
                        args=args,
                        participant_perm_dicts=participant_perm_dicts,
                        simulated_participant=simulated_participant,
                        llm_generator=llm_generator,
                        construct_messages=construct_messages,
                        previous_messages=running_messages,
                        answers=answers,
                        label_2_text_option_dict=label_2_text_option_dict,
                        prompt=prompt,
                        chosen_value=chosen_value,
                    )

                    cors[chosen_item_i] = cor
                    all_lprobs[chosen_item_i] = lprobs
                    all_probs[chosen_item_i] = probs
                    all_answers[chosen_item_i] = pred
                    all_generations[chosen_item_i] = generation
                    all_scores[chosen_item_i] = score


            # Rate the Non-extreme values (it was not already rated)
            if all_scores[item_i] is None:
                cor, lprobs, probs, pred, generation, score, running_messages = score_non_extreme_value_svs(
                    test_df=test_df, item_i=item_i, args=args,
                    previous_messages=running_messages,
                    first_non_extreme_value=first_non_extreme_value,
                    llm_generator=llm_generator,
                    permutations_dict=permutations_dict,
                    format_example=format_example, construct_messages=construct_messages,
                    label_2_text_option_dict=label_2_text_option_dict,
                    answers=answers,
                    simulated_participant=simulated_participant,
                )
                first_non_extreme_value = False

                cors[item_i] = cor
                all_lprobs[item_i] = lprobs
                all_probs[item_i] = probs
                all_answers[item_i] = pred
                all_generations[item_i] = generation
                all_scores[item_i] = score

        else:
            messages = construct_messages(
                prompt=prompt,
                system_message=llm_generator.system_message,
                messages_conv=messages_conv
            )

            generation, lprobs = llm_generator.predict(
                messages=messages,
                answers=answers,
                label_2_text_option_dict=label_2_text_option_dict,
                query_string=prompt['query_str'],
                assistant_label=simulated_participant["name"].upper()
            )

            probs = softmax(np.array(lprobs))
            pred = {i: c for i, c in enumerate(answers)}[np.argmax(lprobs)]
            cor = pred == label
            score = map_choice_to_number(pred, permutations_dict)

            if args.verbose:
                print(colored(f"Pred:{pred} (Generation:{generation}; Score: {score})", "green"))
                print("------------------")

            cors[item_i] = cor
            all_lprobs[item_i] = lprobs
            all_probs[item_i] = probs
            all_answers[item_i] = pred
            all_generations[item_i] = generation
            all_scores[item_i] = score

    cors = np.array(cors)
    all_scores = np.array(all_scores)

    return cors, all_probs, all_lprobs, all_answers, all_scores, all_generations, gpt_token_counter


def main(args):
    model_config_path = args.model_config_path
    print("Model:", model_config_path)

    subjects = sorted([f.split("_test.csv")[0] for f in os.listdir(os.path.join(args.data_dir, "test")) if "_test.csv" in f])

    # add timestamp to dir_name
    dump_results_dir = Path(args.save_dir)
    dump_results_dir = dump_results_dir.with_name(dump_results_dir.name+f"_{timestamp}")

    if not args.overwrite:
        prev_jsons = list(dump_results_dir.parent.glob(dump_results_dir.name.replace(timestamp, "")+"*/results.json"))
        if len(prev_jsons) > 0:
            raise RuntimeError(f"Previous version of this run ({dump_results_dir}) were found: {prev_jsons}")

    else:
        old_jsons = list(Path(dump_results_dir).parent.glob(f"*{args.permute_options_seed}*{args.simulated_conversation_theme}*/results.json"))

        for old_json in old_jsons:
            new_json = old_json.parent / "results_old.json.backup"
            old_json.rename(new_json)
            print(f"Renamed: {old_json} --> {new_json}")

    os.makedirs(dump_results_dir, exist_ok=True)
    print("Savedir: ", dump_results_dir)

    # Data preparation
    if len(subjects) == 0:
        raise ValueError("No subjects found.")

    if "data_pvq" in args.data_dir:
        assert "pvq" in args.experiment_name
        subjects = ["pvq_auto"]

    print("Args:", args)
    print("Subjects:", subjects)

    gpt_tokens_total = {"input": 0, "output": 0}

    if "pvq" in args.data_dir:
        max_n_options = 6
    elif "svs" in args.data_dir:
        max_n_options = 9
    elif "donation" in args.data_dir:
        max_n_options = 6
    elif "bag" in args.data_dir:
        max_n_options = 6
    elif "religion" in args.data_dir:
        max_n_options = 5
    else:
        raise ValueError(f"Undefined number of options for data in {args.data_dir}.")

    if Path(args.simulated_population_config).is_file():
        with open(args.simulated_population_config, 'r') as f:
            simulated_population = json.load(f)

    elif args.simulated_population_config == "permutations":
        simulated_population_genders = (["M", "F"]*int(np.ceil(args.permutations/2)))[:args.permutations]
        simulated_population = [{
            "name": "CHATBOT",
            "description": None,
            "gender": g
        } for g in simulated_population_genders]
    else:
        raise ValueError(f"Undefined population {args.simulated_population_config} - give path to a config file.")

    llm_generator = create_model(model_config_path)
    all_cors = []

    # list because of permutations
    subj_lprobs = [{} for _ in range(len(simulated_population))]
    subj_len = [{} for _ in range(len(simulated_population))]
    metrics = [{} for _ in range(len(simulated_population))]
    answers = [{} for _ in range(len(simulated_population))]
    generations = [{} for _ in range(len(simulated_population))]

    # evaluate model
    for subject in subjects:

        if subject == "pvq_auto":
            test_df_dict = {}
            test_df_dict["F"] = pd.read_csv(
                os.path.join(args.data_dir, args.eval_set, f"pvq_female_{args.eval_set}.csv"),
                header=None, keep_default_na=False,
            )

            test_df_dict["M"] = pd.read_csv(
                os.path.join(args.data_dir, args.eval_set, f"pvq_male_{args.eval_set}.csv"),
                header=None, keep_default_na=False,
            )

            # if the question contains \n in the csv it will get parsed as \\n, we revert it back here to be newline
            test_df_dict["F"][0][:] = test_df_dict["F"][0][:].str.replace("\\n", "\n")
            test_df_dict["M"][0][:] = test_df_dict["M"][0][:].str.replace("\\n", "\n")

            assert len(test_df_dict["F"]) == len(test_df_dict["M"])
            assert test_df_dict["F"].shape == test_df_dict["M"].shape

            num_questions = len(test_df_dict["F"])
            assert max_n_options == test_df_dict["F"].shape[1] - 2
            n_options = [max_n_options] * num_questions

        else:

            test_df = pd.read_csv(
                os.path.join(args.data_dir, args.eval_set, subject + f"_{args.eval_set}.csv"),
                header=None,
                keep_default_na=False,
                dtype=str
            )
            n_options = [max_n_options]*len(test_df)

            # if the question contains \n in the csv it will get parsed as \\n, we revert it back here to be newline
            test_df[0][:] = test_df[0][:].str.replace("\\n", "\n")

            num_questions = len(test_df)

        permutations_dicts = create_permutation_dicts(
            args,
            n_options,
            choices,
            num_questions=num_questions,
            population_size=len(simulated_population)
        )

        assert len(permutations_dicts) == len(simulated_population)
        assert all([len(part_d) == num_questions for part_d in permutations_dicts])

        # Get the topic conversation starter
        opening_questions, per_participant_contexts = get_opening_question_for_theme(
            conversation_theme=args.simulated_conversation_theme
        )

        interlocutors, per_participant_interlocutors = get_interlocutors(interlocutors=args.interlocutors)

        pop_start_time = time.time()

        # evaluate over population
        for sim_part_i, (simulated_participant, participant_perm_dicts) in enumerate(zip(simulated_population, permutations_dicts)):

            if sim_part_i > 0:
                eta = estimate_eta(start_time=pop_start_time, progress=sim_part_i/len(simulated_population))
                eta_str = "ETA: {:.0f}h {:.0f}m {:.2f}s".format(*secs_2_hms(eta))

            else:
                eta_str = ""

            print(f"Simulated participant {sim_part_i}/{len(simulated_population)} {eta_str}")

            if subject == "pvq_auto":
                test_df = test_df_dict[simulated_participant['gender']]

            cors, eval_probs, eval_lprobs, preds, preds_values, gens, gpt_tokens = eval(
                args=args,
                test_df=test_df,
                participant_perm_dicts=participant_perm_dicts,
                llm_generator=llm_generator,
                simulated_participant=simulated_participant,
                opening_question=opening_questions[sim_part_i] if per_participant_contexts else opening_questions,
                interlocutor=interlocutors[sim_part_i] if per_participant_interlocutors else interlocutors
            )

            if args.estimate_gpt_tokens:
                estimate_and_print_gpt_prices(gpt_tokens, args.engine)

            all_cors.append(cors)
            gpt_tokens_total['input'] += gpt_tokens['input']
            gpt_tokens_total['output'] += gpt_tokens['output']

            subj_lprobs[sim_part_i][subject] = eval_lprobs
            subj_len[sim_part_i][subject] = num_questions
            answers[sim_part_i][subject] = list(zip(preds, map(int, preds_values)))
            generations[sim_part_i][subject] = gens

            if "pvq" in args.data_dir or "svs" in args.data_dir:
                assert "pvq" in args.experiment_name or "svs" in args.experiment_name

                profile_values_idx_json = os.path.join(os.path.join(args.data_dir, "raw"), "values.json")

                with open(profile_values_idx_json) as f:
                    profile_values_idx = json.load(f)

                profile_values_idx = {k: np.array(v)-1 for k, v in profile_values_idx.items() if k != "_comment"}

                metrics[sim_part_i][subject] = {}

                for profile_value, idxs in profile_values_idx.items():
                    metrics[sim_part_i][subject][profile_value] = preds_values[idxs].mean()

            elif "tolkien_donation" in args.data_dir:
                assert "donation" in args.experiment_name

                groups = ["elves", "dwarves", "orcs", "humans", "hobbits"]

                donated = (preds_values-1)*2
                group_donations = np.split(donated, len(groups))
                assert set([len(g) for g in group_donations]) == {20}

                metrics[sim_part_i][subject] = {
                    f"Donation {g}": np.mean(g_d) for g, g_d in zip(groups, group_donations)
                }

            elif "tolkien_bag" in args.data_dir:
                assert "bag" in args.experiment_name

                groups = ["elves", "dwarves", "orcs", "humans", "hobbits"]
                group_bag = np.split(preds_values, len(groups))
                assert set([len(g) for g in group_bag]) == {20}

                metrics[sim_part_i][subject] = {
                    f"Return {g}": np.mean(g_d) for g, g_d in zip(groups, group_bag)
                }

            elif "religion" in args.data_dir:
                assert "religion" in args.experiment_name

                metrics[sim_part_i][subject] = {
                    f"religion time": np.mean(preds_values)
                }

            else:
                raise NotImplementedError(f"Evaluation not implemented for {args.data_dir}")

        # assert the same and take the fist
        assert all(subj_len[0] == s for s in subj_len)
        subj_len = subj_len[0]

        # remap from list of metrics to metrics with lists
        mean_metrics = defaultdict(lambda: defaultdict(list))
        for metrics_perm in metrics:
            for subj, subj_metrics in metrics_perm.items():
                for metric, value in subj_metrics.items():
                    mean_metrics[subj][metric].append(value)

        # average metrics
        mean_metrics = {
            subj: {
                metric: np.mean(values) for metric, values in subj_metrics.items()
            } for subj, subj_metrics in mean_metrics.items()
        }

        pop_metrics = {}

        # save results
        for subj, m in mean_metrics.items():
            if m:
                print("Subject: ", subj)
                for metric, score in m.items():
                    print(f"{metric} : {score}")

        if not os.path.exists(dump_results_dir):
            os.mkdir(dump_results_dir)

        json_dump_path = os.path.join(dump_results_dir, 'results.json')

        with open(json_dump_path, 'w') as fp:
            json.dump({
                "args": vars(args),
                "metrics": mean_metrics,
                "pop_metrics": pop_metrics,
                "per_permutation_metrics": metrics,  # legacy todo: remove and update var_viz
                "per_simulated_participant_metrics": metrics,
                "simulated_population": simulated_population,
                "generations": generations,
                "answers": answers,
                "lprobs": subj_lprobs,
                **{
                    "params": vars(args)
                }
            }, fp, indent=4)

        print(f"Results saved to {json_dump_path}")

        if pop_metrics:
            print("pop metrics:", pop_metrics['all']['hist'])

    if args.estimate_gpt_tokens:
        estimate_and_print_gpt_prices(gpt_tokens_total, args.engine)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", "-d", type=str, required=True)
    parser.add_argument("--save_dir", "-s", type=str, default="results/results_test")
    parser.add_argument("--experiment_name", "-n", type=str, default="")
    parser.add_argument("--pvq-version", type=str, default="pvq_auto", choices=["pvq_auto"])
    parser.add_argument("--engine", "-e", type=str, default="dummy")
    parser.add_argument("--model-config-path", type=str, default=None)
    parser.add_argument("--format", type=str, default="chat", choices=["chat"])
    parser.add_argument('--profile', type=str, help='Profile definition in format "k:v;k:v;k:v", ex. "age:35;interests:reading books"')
    parser.add_argument("--query-prompt", "-qp", type=str, help="Use Answer(as ONE letter): where applicable.")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--assert-params", action="store_true")
    parser.add_argument("--estimate-gpt-tokens", "-t", action="store_true")
    parser.add_argument("--eval-set", type=str, default="test", choices=["test", "val"])
    parser.add_argument("--simulated-conversation-theme", type=str, default=None)
    parser.add_argument("--simulated-conversation-n-messages", type=int, default=5)
    parser.add_argument("--interlocutors", type=str, default="human")
    parser.add_argument("--long-messages", action="store_true")
    parser.add_argument("--interlocutor-knows-persona", action="store_true")
    parser.add_argument("--simulated-population-config", "-pop", type=str, required=True)
    parser.add_argument("--permutations", "-p", type=int, default=50)
    parser.add_argument("--permute-options-seed", type=str)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    assert args.pvq_version == "pvq_auto"

    if args.model_config_path is not None:
        # engine is the name of the json file
        model_config_filename = os.path.basename(args.model_config_path)
        args.engine = os.path.splitext(model_config_filename)[0]

    else:
        # use the default config directory
        args.model_config_path = f'./models/configs/{args.engine}.json'

    if not args.data_dir.startswith("data"):
        raise ValueError(f"data_dir should be inside data, and it's {args.data_dir}")

    if args.simulated_population_config == "permutations":
        if args.interlocutor_knows_persona:
            warnings.warn("interlocutor_knows_persona cannot be used with permutations sim. population type -> setting to false.")
            args.interlocutor_knows_persona = False

    if args.simulated_conversation_theme in ["None", "none"]:
        args.simulated_conversation_theme = None

    if args.estimate_gpt_tokens:
        if "gpt" not in args.engine and args.engine != "dummy":
            raise ValueError("Only gpt-4 gpt-3 and dummy support estimating GPT tokens")

    start_time = time.time()
    main(args)
    end_time = time.time()
    print("Elapsed time:", str(datetime.timedelta(seconds=end_time-start_time)).split(".")[0])