import argparse
import random
from collections import defaultdict
from pathlib import Path
import json
import hashlib
import time
import datetime
import itertools
import string

from termcolor import colored

from utils import *

from models import *

import numpy as np
import pandas as pd
import torch
import tiktoken

from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList, BitsAndBytesConfig
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training

from personas.utils import simulated_participant_to_name


hf_cache_dir = get_hf_cache_dir()
os.environ['HF_HOME'] = hf_cache_dir


# take the theme starter
opening_questions_for_themes = {
    "poem": "Hello, let's write a poem together. You start by the first verse I'll add the second one, and so on.",
    "joke": "Tell me a joke.",
    "history": "What is the significance of the battle of Hastings. Answer in two sentences.",  # slight collapse
    "chess": "1. e4",
    "grammar": "Can you check this sentence for grammar? \n Whilst Jane was waiting to meet hers friend their nose started bleeding.",
}


def create_permutation_dicts(args, n_options, choices, num_questions, population_size=None):

    if args.permute_options:

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

    else:
        if len(set(n_options)) == 1:
            permutations_dicts = [
                [{choices[i]: i for i, c in enumerate(choices[:n_opt])} for n_opt in n_options]
            ] * population_size

    return permutations_dicts


def create_simulated_messages(conv, last="user"):
    # simulate a conversation between two LLMs
    if last == "user":
        # last role is user
        sim_conv = list(zip(["user", "assistant"] * (len(conv) // 2 + 1), conv[::-1]))[::-1]
    elif last == "assistant":
        # last role is assistant
        sim_conv = list(zip(["assistant", "user"] * (len(conv) // 2 + 1), conv[::-1]))[::-1]
    else:
        raise ValueError("last must be either user or assistant")

    sim_conv_messages = [{"role": role, "content": msg} for role, msg in sim_conv]

    return sim_conv_messages


class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops, tokenizer, original_input_ids):
        super().__init__()
        self.stops = [s.upper() for s in stops]
        self.tokenizer = tokenizer
        self.original_input_ids = original_input_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs):
        generated_ids = input_ids[0][len(self.original_input_ids[0]):]
        generation = self.tokenizer.decode(generated_ids).upper()
        return any([stop in generation for stop in self.stops])


def simulate_conversation(args, engine, sim_engine, model_set_persona_string=None, llm_generator=None, simulated_participant=None):

    opening_question = opening_questions_for_themes[args.simulated_conversation_theme]

    conversation = [opening_question]

    # simulate conversation
    assert args.simulated_conversation_n_messages % 2 == 1  # must be odd so that the last one is GPT as simulated persona

    for msg_i in range(args.simulated_conversation_n_messages):
        if args.verbose:
            print(f"Simulated conv msg {msg_i}")

        # assign roles to messages - alternating, last one user
        simulated_conv_messages = create_simulated_messages(conversation, last="user")
        simulated_participant_name = simulated_participant_to_name(simulated_participant, args.simulated_population_type)
        labels_dict = {
            "persona": {
                "assistant_label": simulated_participant_name.upper(),
                "user_label": "USER",
                "system_label": "CONTEXT"
            },
            "human": {
                "assistant_label": "HUMAN",
                "user_label": f"{simulated_participant_name.upper()} (CHATBOT)" if args.simulated_human_knows_persona else "CHATBOT",
                "system_label": "CONTEXT"
            }
        }
        stop_words_up = [f"\n{v}:" for v in labels_dict["persona"].values()] + [f"\n{v}:" for v in labels_dict["human"].values()]
        # also add similar words wo whitespace ex. GANDALF (CHATBOT) and GANDALF(CHATBOT)
        stop_words_up += [s.replace(" ", "") for s in stop_words_up if " " in s]

        if msg_i % 2 == 0:
            # even -> gpt as a persona
            assert simulated_conv_messages[0]['role'] == "user"

            if model_set_persona_string:
                simulated_conv_messages = [{
                    "role": "system" if llm_generator.system_message else "user",
                    "content": model_set_persona_string
                }] + simulated_conv_messages

            engine_ = engine
            assistant_label = labels_dict["persona"]["assistant_label"]
            user_label = labels_dict["persona"]["user_label"]
            system_label = labels_dict["persona"]["system_label"]

        else:
            # gpt as human
            assert simulated_conv_messages[0]['role'] == "assistant"

            # user doesn't know the chatbots persona -> change this?
            # if args.base_model_template:
            if llm_generator.base_model_template:
                if args.simulated_human_knows_persona:
                    sys_msg = f"The following is a conversation between a human and a chatbot. The chatbot is pretending to be {simulated_participant_name}. The human's every reply must be in one sentence only."
                else:
                    sys_msg = f"The following is a conversation between a human and a chatbot. The human's every reply must be in one sentence only."
            else:
                if args.simulated_human_knows_persona:
                    sys_msg = f"You are simulating a human using a chatbot. The chatbot is pretending to be {simulated_participant_name}. Your every reply must be in one sentence only."
                else:
                    sys_msg = f"You are simulating a human using a chatbot. Your every reply must be in one sentence only."

            simulated_conv_messages = [{
                "role": "system" if llm_generator.system_message else "user",
                "content": sys_msg
            }] + simulated_conv_messages

            assistant_label = labels_dict["human"]["assistant_label"]
            user_label = labels_dict["human"]["user_label"]
            system_label = labels_dict["human"]["system_label"]

        # if not args.base_model_template:
        if not llm_generator.base_model_template:
            simulated_conv_messages = fix_alternating_msg_order(simulated_conv_messages)

        response = llm_generator.generate(
            messages=simulated_conv_messages,
            assistant_label=assistant_label,
            user_label=user_label,
            system_label=system_label,
            stop_words_up=stop_words_up
        )

        if args.verbose:
            print_chat_messages(simulated_conv_messages)


        # llm_generator_type = type(llm_generator)
        # if llm_generator_type == HuggingFaceModel:
        #     response = llm_generator.generate(
        #         messages=simulated_conv_messages,
        #         generation_kwargs=dict(
        #             max_new_tokens=args.simulated_conversation_msg_max_tokens,
        #             do_sample=True,
        #             top_p=args.simulated_conversation_top_p,
        #             temperature=args.simulated_conversation_temp,
        #             # top_k=50,
        #             # repetition_penalty=1.2,  # logit / (T * penalty*bool(token present) )
        #             num_beams=1,
        #         ),
        #         assistant_label=assistant_label,
        #         user_label=user_label,
        #         system_label=system_label,
        #         stop_words_up=stop_words_up
        #     )
        #
        # elif llm_generator_type == OpenAIModel:
        #     response = llm_generator.generate(
        #         messages=simulated_conv_messages,
        #         generation_kwargs=dict(
        #             max_tokens=args.simulated_conversation_msg_max_tokens,
        #             top_p=args.simulated_conversation_top_p,
        #             temperature=args.simulated_conversation_temp,
        #             # not the same as hf repetition_penalty
        #             # presence_penalty=0.2,  # logit - penalty*bool(token present)
        #             n=1,
        #         )
        #     )
        # elif llm_generator_type in [InteractiveModel, DummyModel]:
        #     response = llm_generator.generate()
        #
        # else:
        #     raise NotImplementedError(f"Simulated conversations not implemented for {engine_}")

        # if args.base_model_template:
        if llm_generator.base_model_template:
            response_up = response.upper()
            stop_word_ind = np.min([response_up.index(sw) if sw in response_up else np.inf for sw in stop_words_up])
            if stop_word_ind != np.inf:
                stop_word_ind = int(stop_word_ind)
                response = response[:stop_word_ind]

        conversation.append(response)

        if args.verbose:
            print(f"--> {response}")

        messages_conv = create_simulated_messages(conversation, last="assistant")
        messages_conv_hash = hash_chat_conv(messages_conv)

    return messages_conv, messages_conv_hash


def map_choice_to_number(letter, permutations_dict):
    # A-F -> 1-6
    # find index of letter in choices and add 1
    number = permutations_dict[letter] + 1
    return number


def map_number_to_choice(number, inv_permutations_dict):
    choice = inv_permutations_dict[number-1]
    return choice


timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
print("timestamp:", timestamp)


hf_token = os.environ["HF_TOKEN"]
hidden_token = hf_token[:6] + "*" * (len(hf_token)-12) + hf_token[-6:]
print("HF TOKEN:", hidden_token)


# choices = ["A", "B", "C", "D", "E", "F"]
choices = list(string.ascii_uppercase)

llama_dir = "/gpfswork/rech/imi/utu57ed/llama/llama_files/"


def get_prompt_skeleton(experiment_name, args, simulated_participant, base_model_template):

    if "pvq" in experiment_name:
        assert "pvq" in args.data_dir
        questionnaire_description = "Here we briefly describe some people. Please read each description and think about how much each person is or is not like you. Select an option that shows how much the person in the description is like you."
        questionnaire_description_empty = False

    elif "donation" in experiment_name:
        assert "donation" in args.data_dir
        questionnaire_description = ""
        questionnaire_description_empty = True

    elif "bag" in experiment_name:
        assert "bag" in args.data_dir
        questionnaire_description = ""
        questionnaire_description_empty = True

    elif "religion" in experiment_name:
        assert "religion" in args.data_dir
        questionnaire_description = ""
        questionnaire_description_empty = True

    else:
        raise ValueError(f"Experiment name is ill-defined {args.experiment_name}")

    if base_model_template:
        prefix = "The following is a conversation with"
    else:
        prefix = "You are"

    if args.simulated_population_type in ["famous_people"]:
        set_persona_str = f"{prefix} {simulated_participant}"
    elif args.simulated_population_type in ["tolkien_characters"]:
        set_persona_str = f"{prefix} {simulated_participant} from J. R. R. Tolkien's Middle-earth legendarium."
    elif args.simulated_population_type == "permutations":
        set_persona_str = ""
    else:
        raise ValueError("Unknown population type")

    if args.query_prompt:
        query_str = args.query_prompt
    else:
        query_str = "Answer: ("

    assert (not questionnaire_description_empty == questionnaire_description)

    prompt_skeleton = {
        "set_persona_str": set_persona_str,  # remove newline from the end
        "questionnaire_description": questionnaire_description,
        "query_str": f"{query_str}",
    }

    return prompt_skeleton

def format_example(df, idx, experiment_name, args, permutations_dict, simulated_participant, base_model_template=None):
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

    for ch in choices[:num_options]:
        item_str += "\n({}) {}".format(ch, options_strings[permutations_dict[ch]])

    prompt = get_prompt_skeleton(
        experiment_name=experiment_name,
        args=args,
        simulated_participant=simulated_participant,
        base_model_template=base_model_template
    )

    prompt["item_str"] = item_str

    return prompt, num_options


def hash_chat_conv(msgs_conv):
    json_string = json.dumps(msgs_conv)

    # Create a SHA256 hash of the string
    hash_object = hashlib.sha256(json_string.encode())

    # Get the hexadecimal representation of the hash
    hex_dig = hash_object.hexdigest()

    return hex_dig


def eval(args, engine, test_df, participant_perm_dicts, llm_generator=None, simulated_participant=None):
    cors = []
    all_probs = []
    all_lprobs = []
    all_answers = []
    all_generations = []
    all_scores = []

    # hashing for simulated conversations
    messages_conv = None
    messages_conv_hash = None

    gpt_token_counter = {"input": 0, "output": 0}

    assert test_df.shape[0] == len(participant_perm_dicts)

    for item_i, permutations_dict in enumerate(participant_perm_dicts):
        inv_permutations_dict = {v: k for k, v in permutations_dict.items()}

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
            simulated_participant=simulated_participant,
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

        if args.simulated_conversation_theme:

            set_persona_str = prompt["set_persona_str"]
            if messages_conv is None:
                if args.verbose:
                    print("SIMULATING CONVERSATION")

                messages_conv, messages_conv_hash = simulate_conversation(
                    args=args,
                    engine=engine,
                    sim_engine=engine,
                    model_set_persona_string=set_persona_str,
                    simulated_participant=simulated_participant,
                    llm_generator=llm_generator,
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
                if args.verbose:
                    print("LOADING CACHED CONVERSATION")
                assert hash_chat_conv(messages_conv) == messages_conv_hash

        if args.estimate_gpt_tokens:
            # gpt params
            messages = construct_messages(
                prompt=prompt,
                system_message=True,
                messages_conv=messages_conv if args.simulated_conversation_theme else None,
            )
            n_input_tokens = sum([len(gpt_tokenizer.encode(msg['content'])) for msg in messages])

            gpt_token_counter['input'] += n_input_tokens
            gpt_token_counter['output'] += 1

        messages = construct_messages(
            prompt=prompt,
            system_message=llm_generator.system_message,
            messages_conv=messages_conv if args.simulated_conversation_theme else None,
        )

        if args.verbose:
            print_chat_messages(messages)

        generation, lprobs = llm_generator.predict(
            messages=messages,
            answers=answers,
            label_2_text_option_dict=label_2_text_option_dict,
            query_string=prompt['query_str'],
            assistant_label=simulated_participant_to_name(simulated_participant, args.simulated_population_type).upper()
        )

        probs = softmax(np.array(lprobs))
        pred = {i: c for i, c in enumerate(answers)}[np.argmax(lprobs)]
        cor = pred == label
        score = map_choice_to_number(pred, permutations_dict)

        if args.verbose:
            print(colored(f"Pred:{pred} (Generation:{generation}; Score: {score})", "green"))
            print("------------------")

        cors.append(cor)
        all_lprobs.append(lprobs)
        all_probs.append(probs)
        all_answers.append(pred)
        all_generations.append(generation)
        all_scores.append(score)

    cors = np.array(cors)
    all_scores = np.array(all_scores)

    if args.estimate_gpt_tokens:
        estimate_and_print_gpt_prices(gpt_token_counter, engine)

    return cors, all_probs, all_lprobs, all_answers, all_scores, all_generations, gpt_token_counter


def main(args):
    engine = args.engine
    print("Engine:", engine)

    subjects = sorted([f.split("_test.csv")[0] for f in os.listdir(os.path.join(args.data_dir, "test")) if "_test.csv" in f])

    # add timestamp to dir_name
    dump_results_dir = Path(args.save_dir)
    dump_results_dir = dump_results_dir.with_name(dump_results_dir.name+f"_{timestamp}")

    if not args.overwrite:
        prev_jsons = list(dump_results_dir.parent.glob(dump_results_dir.name.rstrip(timestamp)+"*/results.json"))
        if len(prev_jsons) > 0:
            raise RuntimeError(f"Previous version of this run were found: {prev_jsons}")

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

        # assert set(subjects_to_evaluate).issubset(subjects)
        subjects = ["pvq_auto"]

    print("Args:", args)
    print("Subjects:", subjects)

    gpt_tokens_total = {"input": 0, "output": 0}

    llm_generator = create_model(
        engine,
        additional_model_args={"use_azure": args.azure_openai}
    )

    if "pvq" in args.data_dir:
        max_n_options = 6
    elif "donation" in args.data_dir:
        max_n_options = 6
    elif "bag" in args.data_dir:
        max_n_options = 6
    elif "religion" in args.data_dir:
        max_n_options = 5
    else:
        raise ValueError(f"Undefined number of options for data in {args.data_dir}.")

    if args.simulated_population_type == "permutations":
        simulated_population = [None]*args.permutations
        simulated_population_genders = (["M", "F"]*int(np.ceil(args.permutations/2)))[:args.permutations]

    elif args.simulated_population_type == "tolkien_characters":
        # https://en.wikipedia.org/wiki/List_of_Middle-earth_characters
        # 50 characters with the longest wikipedia page
        with open("personas/tolkien_characters/tolkien_characters.txt") as f:
            simulated_population = [name.rstrip() for name in f.readlines()]

        with open("personas/tolkien_characters/tolkien_characters_genders.txt") as f:
            simulated_population_genders = [g.rstrip() for g in f.readlines()]

    elif args.simulated_population_type == "famous_people":
        # source: https://www.biographyonline.net/people/famous-100.html
        with open("personas/famous_people/famous_people.txt") as f:
            simulated_population = [name.rstrip() for name in f.readlines()]

        with open("personas/famous_people/famous_people_genders.txt") as f:
            simulated_population_genders = [g.rstrip() for g in f.readlines()]

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
            if not simulated_population_genders:
                raise ValueError("Simulated population genders is not defined.")

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

        pop_start_time = time.time()
        # evaluate over population
        for sim_part_i, (simulated_participant, simulated_participant_gender, participant_perm_dicts) in enumerate(zip(simulated_population, simulated_population_genders, permutations_dicts)):

            if sim_part_i > 0:
                eta = estimate_eta(start_time=pop_start_time, progress=sim_part_i/len(simulated_population))
                eta_str = "ETA: {:.0f}h {:.0f}m {:.2f}s".format(*secs_2_hms(eta))

            else:
                eta_str = ""

            print(f"Simulated participant {sim_part_i}/{len(simulated_population)} {eta_str}")

            if subject == "pvq_auto":
                test_df = test_df_dict[simulated_participant_gender]

            cors, eval_probs, eval_lprobs, preds, preds_values, gens, gpt_tokens = eval(
                args=args,
                engine=engine,
                test_df=test_df,
                participant_perm_dicts=participant_perm_dicts,
                llm_generator=llm_generator,
                simulated_participant=simulated_participant,
            )
            all_cors.append(cors)
            gpt_tokens_total['input'] += gpt_tokens['input']
            gpt_tokens_total['output'] += gpt_tokens['output']

            subj_lprobs[sim_part_i][subject] = eval_lprobs
            subj_len[sim_part_i][subject] = num_questions
            answers[sim_part_i][subject] = list(zip(preds, map(int, preds_values)))
            generations[sim_part_i][subject] = gens

            if "pvq" in args.data_dir:
                assert "pvq" in args.experiment_name

                profile_values_idx_json = os.path.join(os.path.join(args.data_dir, "raw"), "values.json")

                with open(profile_values_idx_json) as f:
                    profile_values_idx = json.load(f)

                profile_values_idx = {k: np.array(v)-1 for k, v in profile_values_idx.items() if k != "_comment"}

                metrics[sim_part_i][subject] = {}

                for profile_value, idxs in profile_values_idx.items():
                    metrics[sim_part_i][subject][profile_value] = preds_values[idxs].mean() # legacy: todo: remove and save those below

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
                raise NotImplementedError("Evaluation not implemented")

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
        estimate_and_print_gpt_prices(gpt_tokens_total, engine)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", "-d", type=str, required=True)
    parser.add_argument("--save_dir", "-s", type=str, default="results/results_test")
    parser.add_argument("--experiment_name", "-n", type=str, default="")
    parser.add_argument("--pvq-version", type=str, default="pvq_auto", choices=["pvq_auto"])
    parser.add_argument("--engine", "-e", type=str, default="dummy")
    parser.add_argument("--format", type=str, default="chat", choices=["chat"])
    parser.add_argument('--profile', type=str, help='Profile definition in format "k:v;k:v;k:v", ex. "age:35;interests:reading books"')
    parser.add_argument("--query-prompt", "-qp", type=str, help="Use Answer(as ONE letter): where applicable.")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--assert-params", action="store_true")
    parser.add_argument("--estimate-gpt-tokens", "-t", action="store_true")
    parser.add_argument("--eval-set", type=str, default="test", choices=["test", "val"])
    # parser.add_argument("--simulated-conversation-msg-max-tokens", type=int, default=100)
    # parser.add_argument("--simulated-conversation-top-p", type=float, default=0.9)
    # parser.add_argument("--simulated-conversation-temp", type=float, default=0.7)
    parser.add_argument("--simulated-conversation-theme", type=str, default=None)
    parser.add_argument("--simulated-conversation-n-messages", type=int, default=5)
    parser.add_argument("--permute-options", "-po", action="store_true")
    parser.add_argument("--azure-openai", action="store_true")
    parser.add_argument("--simulated-human-knows-persona", action="store_true")
    parser.add_argument("--simulated-population-type", "-pop", type=str, default="tolkien_characters", choices=["permutations", "tolkien_characters", "famous_people", "llm_personas", "user_personas", "anes"])
    parser.add_argument("--permutations", "-p", type=int, default=1)  # permutations as a population type
    parser.add_argument("--permute-options-seed", type=str)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    assert args.pvq_version == "pvq_auto"

    if not args.data_dir.startswith("data"):
        raise ValueError(f"data_dir should be inside data, and it's {args.data_dir}")

    if args.simulated_population_type == "permutations":
        if args.simulated_human_knows_persona:
            raise ValueError("Use simulated_human_knows_persona cannot be used with permutations sim. population type")
    else:
        if args.simulated_conversation_theme and not args.simulated_human_knows_persona:
            raise ValueError("Use simulated_human_knows_persona.")

    if args.simulated_conversation_theme in ["None", "none"]:
        args.simulated_conversation_theme = None

    if args.estimate_gpt_tokens:
        if "gpt" not in args.engine and args.engine != "dummy":
            raise ValueError("Only gpt-4 gpt-3 and dummy support estimating GPT tokens")

    if args.permute_options and args.permute_options_seed is None:
        raise ValueError("Permute options string should be defined for stability")

    if ("gpt-3.5" in args.engine and args.permutations > 50) or ("gpt-4" in args.engine and args.permutations > 5):
        raise ValueError(f"Are you sure you want to use {args.permutations} with {args.engine}??")

    start_time = time.time()
    main(args)
    end_time = time.time()
    print("Elapsed time:", str(datetime.timedelta(seconds=end_time-start_time)).split(".")[0])