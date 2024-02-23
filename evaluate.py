import argparse
import datetime
import random
import re
import json
from collections import defaultdict
import os

import matplotlib.pyplot as plt
import tiktoken

import torch
import time
import datetime
import itertools

from personas.utils import simulated_participant_to_name

from termcolor import colored

hostname = os.uname()[1]
if hostname == "PTB-09003439":
    hf_cache_dir = "/home/flowers-user/.cache/huggingface"
elif "plafrim" in hostname:
    hf_cache_dir ="/beegfs/gkovac/hf_cache_dir"
else:
    hf_cache_dir = "/gpfsscratch/rech/imi/utu57ed/.cache/huggingface"

os.environ['TRANSFORMERS_CACHE'] = hf_cache_dir
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList, TextStreamer, pipeline

import json
import hashlib

def construct_messages(prompt, system_message, messages_conv=None, add_query_str=True):

    set_persona_str = prompt["set_persona_str"]
    questionnaire_description = prompt["questionnaire_description"]

    user_prompt = f"{questionnaire_description}\n\n" if questionnaire_description else ""
    user_prompt += prompt["item_str"]

    if add_query_str:
        user_prompt += "\n"+prompt["query_str"]

    if system_message or messages_conv:
        # multiple messages
        messages = []
        if set_persona_str:
            messages.append({
                "role": "system" if system_message else "user",
                "content": set_persona_str
            })

        if messages_conv:
            messages.extend(messages_conv)

        messages.append({"role": "user", "content": user_prompt})

        if not system_message:
            # USER, USER -> USER, AS:"OK", USER
            messages = fix_alternating_msg_order(messages)

    else:

        full_prompt = f"{set_persona_str}\n\n" if set_persona_str else ""

        if args.separator:
            full_prompt += "-" * 200 + "\n"

        full_prompt += user_prompt

        messages = [
            {"role": "user", "content": full_prompt}
        ]

    return messages


def apply_base_model_template(
        messages,
        simulated_participant,
        simulated_population_type,
        assistant_label,
        user_label,
        system_label,
        add_generation_prompt=True,
        return_stop_words=False,
):

    formatted_conversation = ""

    labels_dict = {
        "ASSISTANT": assistant_label,
        "SYSTEM": system_label,
        "USER": user_label,
    }

    assert assistant_label != ""
    assert user_label != ""
    assert system_label != ""

    for msg in messages:
        label = labels_dict[msg['role'].upper()]
        formatted_conversation += f"{label}:{msg['content']}"
        formatted_conversation += "\n"

    if add_generation_prompt:
        formatted_conversation += f"{labels_dict['ASSISTANT']}:"

    if return_stop_words:
        return formatted_conversation, [f"\n{l}:" for l in labels_dict.values()]
    else:
        return formatted_conversation


# take the theme starter
opening_questions_for_themes = {
    "poem": "Hello, let's write a poem together. You start by the first verse I'll add the second one, and so on.",
    "joke": "Tell me a joke.",
    "history": "What is the significance of the battle of Hastings. Answer in two sentences.",  # slight collapse
    "chess": "1. e4",
    "grammar": "Can you check this sentence for grammar? \n Whilst Jane was waiting to meet hers friend their nose started bleeding.",
}


def print_chat_messages(messages):
    print("*********************")
    print("Messages:")
    for msg in messages:
        print(f"{msg['role'].upper()} : {msg['content']}")
    print("*********************")


def extract_answer_tokens(answers, tokenizer):
    answer_tokens = {a: [] for a in answers}
    for tok_ind in range(len(tokenizer)):
        tok = tokenizer.decode([tok_ind])
        if tok in answers:
            answer_tokens[tok].append(tok_ind)

    return answer_tokens


def create_permutation_dicts(args, n_options, choices, num_questions, population_size=None):

    if args.permute_options:

        # sample permutations based on given seed -> should correspond to different contexts
        original_state = random.getstate()  # save the original state
        random.seed(args.permute_options_seed)

        if len(set(n_options)) == 1:

            if n_options[0] > 8:
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


def parse_hf_outputs(output, tokenizer, answers):

    answer_tokens = extract_answer_tokens(answers, tokenizer)  # todo: repetitive -> extract

    option_scores = {
        ans: max([output.scores[0][0, ind] for ind in answer_tokens[ans]])
        for ans in answers
    }

    # take the most probable answer as the generation
    generation = max(option_scores, key=option_scores.get)

    # extract logprobs
    lprobs = [float(option_scores[a]) for a in answers]

    # todo: check that ' A' are one token and check for those as well and not "unk"
    encoded_ans = [tokenizer.encode(ans, add_special_tokens=False)[0] for ans in answers]
    option_scores = {enc_a: output.scores[0][0, enc_a] for enc_a in encoded_ans}

    return option_scores, generation, lprobs


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


def fix_alternating_msg_order(messages):

    if len(messages) <= 1:
        return messages

    # roles must iterate, and start with user, so we add fixes
    if messages[0]['role'] == "system" and messages[1]['role'] == "assistant":
        # insert empty user message
        messages.insert(1, {"role": "user", "content": ""})

    if messages[0]['role'] == "user" and messages[1]['role'] == "user":
        # first message sets the persona, second sets the topic
        # insert artificial message of the model accepting the persona
        messages.insert(1, {"role": "assistant", "content": "OK"})

    return messages


class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops, tokenizer, original_input_ids):
        super().__init__()
        self.stops = [s.upper() for s in stops]
        self.tokenizer = tokenizer
        self.original_input_ids = original_input_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        generated_ids = input_ids[0][len(self.original_input_ids[0]):]
        generation = self.tokenizer.decode(generated_ids).upper()
        return any([stop in generation for stop in self.stops])


def simulate_conversation(args, engine, sim_engine, model_set_persona_string=None, llm_generator=None, simulated_participant=None):

    if llm_generator is not None:
        tokenizer, model = llm_generator

    opening_question = opening_questions_for_themes[args.simulate_conversation_theme]

    conversation = [opening_question]

    # simulate conversation
    assert args.simulated_conversation_n_messages % 2 == 1  # must be odd so that the last one is GPT as simulated persona

    for msg_i in range(args.simulated_conversation_n_messages):
        print(f"Iter {msg_i}")

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
                    "role": "system" if args.system_message else "user",
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
            if args.base_model_template:
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
                "role": "system" if args.system_message else "user",
                "content": sys_msg
            }] + simulated_conv_messages

            engine_ = sim_engine

            assistant_label = labels_dict["human"]["assistant_label"]
            user_label = labels_dict["human"]["user_label"]
            system_label = labels_dict["human"]["system_label"]

        if not args.base_model_template:
            simulated_conv_messages = fix_alternating_msg_order(simulated_conv_messages)

        if engine_ == "dummy":
            response = f"Dummy simulated message no. {msg_i}."

        elif "gpt" in engine_:
            assert not args.base_model_template

            print_chat_messages(simulated_conv_messages)

            c = openai.ChatCompletion.create(
                model=engine_,
                messages=simulated_conv_messages,
                max_tokens=100,
                n=1,
                temperature=1.0,
                request_timeout=30,
            )
            response = c['choices'][0]['message']['content']

        elif "llama_2" in engine_:

            if args.base_model_template:
                assert args.system_message
                formatted_prompt, stop_words = apply_base_model_template(
                    simulated_conv_messages,
                    assistant_label=assistant_label,
                    user_label=user_label,
                    system_label=system_label,
                    simulated_participant=simulated_participant,
                    simulated_population_type=args.simulated_population_type,
                    add_generation_prompt=True,
                    return_stop_words=True
                )
                input_ids = tokenizer(formatted_prompt, return_tensors="pt").to(model.device).input_ids
                assert all([w.upper() in stop_words_up for w in stop_words])
                stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stop_words_up, tokenizer, input_ids)])

                print(f"\n>>>>>>>>>>>>FORMATTED<<<>>>PROMPT<<<<<<<<<<<<\n{formatted_prompt}\n>>>>>>>>>>><<<<<<<<<<<\n")

            else:
                input_ids = tokenizer.apply_chat_template(simulated_conv_messages, return_tensors="pt", add_generation_prompt=True).to(model.device)
                print_chat_messages(simulated_conv_messages)
                stopping_criteria=None

            output_seq = model.generate(
                input_ids=input_ids,
                max_new_tokens=100,
                do_sample=True,
                top_p=0.9,
                top_k=50,
                temperature=0.6,
                repetition_penalty=1.2,
                num_beams=1,
                return_dict_in_generate=True,
                output_scores=True,
                stopping_criteria=stopping_criteria
            )
            response = tokenizer.decode(output_seq.sequences[0][len(input_ids[0]):], skip_special_tokens=True)

        elif engine_ in ["phi-2", "phi-1.5", "phi-1", "Qwen-72B", "Qwen-14B", "Qwen-7B"]:

            if args.base_model_template:
                assert args.system_message
                formatted_prompt, stop_words = apply_base_model_template(
                    simulated_conv_messages,
                    assistant_label=assistant_label,
                    user_label=user_label,
                    system_label=system_label,
                    simulated_participant=simulated_participant,
                    simulated_population_type=args.simulated_population_type,
                    add_generation_prompt=True,
                    return_stop_words=True
                )
                input_ids = tokenizer(formatted_prompt, return_tensors="pt").to(model.device).input_ids
                assert all([w.upper() in stop_words_up for w in stop_words])
                stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stop_words_up, tokenizer, input_ids)])

                print(f"\n>>>>>>>>>>>>FORMATTED<<<>>>PROMPT<<<<<<<<<<<<\n{formatted_prompt}\n>>>>>>>>>>><<<<<<<<<<<\n")

            else:
                input_ids = tokenizer.apply_chat_template(simulated_conv_messages, return_tensors="pt", add_generation_prompt=True).to(model.device)
                print_chat_messages(simulated_conv_messages)
                stopping_criteria=None

            output_seq = model.generate(
                input_ids=input_ids,
                max_new_tokens=100,
                return_dict_in_generate=True,
                output_scores=True,
                stopping_criteria=stopping_criteria
            )
            response = tokenizer.decode(output_seq.sequences[0][len(input_ids[0]):], skip_special_tokens=True)

        elif "zephyr" in engine_ or "Mixtral" in engine_ or "Mistral" in engine_:
            
            # for params: https://huggingface.co/blog/mixtral
            # for params: https://huggingface.co/HuggingFaceH4/zephyr-7b-alpha
            # what about mistral?

            if args.base_model_template:
                assert args.system_message
                formatted_prompt, stop_words = apply_base_model_template(
                    simulated_conv_messages,
                    assistant_label=assistant_label,
                    user_label=user_label,
                    system_label=system_label,
                    simulated_participant=simulated_participant,
                    simulated_population_type=args.simulated_population_type,
                    add_generation_prompt=True,
                    return_stop_words=True
                )
                input_ids = tokenizer(formatted_prompt, return_tensors="pt").to(model.device).input_ids
                assert all([w.upper() in stop_words_up for w in stop_words])
                stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stop_words_up, tokenizer, input_ids)])

                print(f"\n>>>>>>>>>>>>FORMATTED<<<>>>PROMPT<<<<<<<<<<<<\n{formatted_prompt}\n>>>>>>>>>>><<<<<<<<<<<\n")

            else:
                input_ids = tokenizer.apply_chat_template(simulated_conv_messages, return_tensors="pt", add_generation_prompt=True).to(model.device)
                print_chat_messages(simulated_conv_messages)
                stopping_criteria=None

            output_seq = model.generate(
                input_ids=input_ids,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                top_k=50,
                return_dict_in_generate=True,
                output_scores=True,
                stopping_criteria=stopping_criteria
            )
            response = tokenizer.decode(output_seq.sequences[0][len(input_ids[0]):], skip_special_tokens=True)

        else:
            raise NotImplementedError(f"Simulated conversations not implemented for {engine_}")

        if args.base_model_template:
            response_up = response.upper()
            stop_word_ind = np.min([response_up.index(sw) if sw in response_up else np.inf for sw in stop_words_up])
            if stop_word_ind != np.inf:
                stop_word_ind = int(stop_word_ind)
                response = response[:stop_word_ind]

        conversation.append(response)
        print(f"--> {response}")


        messages_conv = create_simulated_messages(conversation, last="assistant")
        messages_conv_hash = hash_chat_conv(messages_conv)

    # print_chat_messages(messages_conv)

    return messages_conv, messages_conv_hash


def map_choice_to_number(letter, permutations_dict):
    # A-F -> 1-6
    # find index of letter in choices and add 1
    number = permutations_dict[letter] + 1
    return number

def map_number_to_choice(number, inv_permutations_dict):
    choice = inv_permutations_dict[number-1]
    return choice


def plot_dict(data, savefile=None):
    # Get the keys and values from the dictionary
    keys = list(data.keys())
    values = list(data.values())

    # Create a bar chart with the keys as the x-axis and the values as the y-axis
    fig, ax = plt.subplots()
    ax.bar(keys, values)

    # Set the title and axis labels
    # ax.set_title('Values by Index')
    ax.set_xlabel('')
    ax.set_ylabel('Values')

    # Rotate the x-axis labels for readability
    plt.xticks(rotation=90)

    plt.tight_layout()
    # Display the plot
    if savefile is not None:
        plt.savefig(savefile)
        print(f'Plot saved to "{savefile}"')
    else:
        plt.show()


timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
print("timestamp:", timestamp)

import openai
import numpy as np
import pandas as pd
import sys

from crop import crop

openai.api_key = os.environ["OPENAI_API_KEY"]
hidden_key = openai.api_key[:10] + "*" * 10 + openai.api_key[20:]
print("OPENAI KEY:", hidden_key)

hf_token = os.environ["HF_TOKEN"]
hidden_token = hf_token[:6] + "*" * (len(hf_token)-12) + hf_token[-6:]
print("HF TOKEN:", hidden_token )


# choices = ["A", "B", "C", "D", "E", "F"]
import string
choices = list(string.ascii_uppercase)

llama_dir = "/gpfswork/rech/imi/utu57ed/llama/llama_files/"

def get_prompt_skeleton(subject, experiment_name, args, simulated_participant):

    if args.format in ["code_py", "code_cpp", "conf_toml", "latex"]:
        if not args.no_profile:
            raise NotImplementedError(f"{args.format} format is only implemented with no_profile.")

    if args.weather:
        if not args.no_profile:
            raise NotImplementedError(f"{args.weather} is only implemented with no_profile.")

    if "pvq" in experiment_name:
        assert "pvq" in args.data_dir
        questionnaire_description = "Here we briefly describe some people. Please read each description and think about how much each person is or is not like you. Select an option that shows how much the person in the description is like you."
        questionnaire_description_empty = False

    elif "ultimatum" in experiment_name:
        assert "ultimatum" in args.data_dir
        questionnaire_description = "In the following scenario, You have to decide whether to accept or reject a proposal."
        questionnaire_description_empty = False

    elif "tolkien_good_bad" in experiment_name:
        assert "tolkien_good_bad" in args.data_dir
        questionnaire_description = ""
        questionnaire_description_empty = True

    elif "donation" in experiment_name:
        assert "donation" in args.data_dir
        questionnaire_description = ""
        questionnaire_description_empty = True

    elif "tolkien_public_goods" in experiment_name:
        assert "tolkien_public_goods" in args.data_dir
        questionnaire_description = ""
        questionnaire_description_empty = True

    elif "wvs_svas" in experiment_name:
        assert "wvs_svas" in args.data_dir
        # VSM questionnaire doesn't have a description
        questionnaire_description = ""
        questionnaire_description_empty = True


    elif "hofstede" in experiment_name:
        assert "hofstede" in args.data_dir
        # VSM questionnaire doesn't have a description
        questionnaire_description = ""
        questionnaire_description_empty = True

    elif "big5" in experiment_name:
        questionnaire_description = "Mark how much you agree with each statement."
        questionnaire_description_empty = False

    elif "mmlu" in experiment_name:
        assert "mmlu" in args.data_dir
        questionnaire_description = "The following are multiple choice questions (with answers)."
        questionnaire_description_empty = False

    else:
        raise ValueError(f"Experiment name is ill-defined {args.experiment_name}")

    assert args.no_profile
    if args.base_model_template:
        prefix = "The following is a conversation with"
    else:
        prefix = "You are"

    if args.simulated_population_type in ["famous_people"]:
        set_persona_str = f"{prefix} {simulated_participant}"
    elif args.simulated_population_type in ["tolkien_characters"]:
        set_persona_str = f"{prefix} {simulated_participant} from J. R. R. Tolkien's Middle-earth legendarium."
    elif args.simulated_population_type == "anes":
        set_persona_str = f"{prefix} a person with the following profile.\n\n'''\n{simulated_participant}'''"
    elif args.simulated_population_type == "llm_personas":
        set_persona_str = f"{prefix} a person with the following profile.\n\n'''\n{simulated_participant}'''"
    elif args.simulated_population_type == "user_personas":
        if args.base_model_tempalte:
            raise NotImplementedError("base model template not implemented for user_personas simulated population.")
        set_persona_str = f"{prefix} talking to a person with the following profile.\n\n'''\n{simulated_participant}'''"
    elif args.simulated_population_type == "permutations":
        set_persona_str = ""
    else:
        raise ValueError("Unknown population type")

    if args.weather:
        weather_dict = {
            "rain": "It is a rainy day.",
            "sun": "It is a beautiful sunny day.",
            "snow": "It is a snowy winter day",
            "thunderstorm": "There is a severe thunderstorm.",
            "sandstorm": "There is a severe sandstorm and a heat wave.",
            "blizzard": "There is a severe blizzard.",
        }
        set_persona_str = set_persona_str.rstrip() + f"\n{weather_dict[args.weather]}"

    if args.format == "code_py":
        query_str = """# Choose the answer\nanswer = answers_dict[\"("""

        questionnaire_description = f"""query = \"\"\"\n{questionnaire_description}"""

    elif args.format == "code_cpp":
        query_str = "\t// Choose the answer\n\tstd::string answer = answers_dict[\"("

        questionnaire_description = "#include <iostream>\n" + \
        "#include <string>\n" + \
        "#include <map>\n" + \
        "int main() {\n" + \
        "\tstd::string query = R\"(\n" + \
        questionnaire_description

    elif args.format == "conf_toml":
        query_str = "answer = ("

        if questionnaire_description_empty:
            questionnaire_description = \
                "[questionnaire]\n"
        else:
            questionnaire_description = \
                "[questionnaire]\n" + \
                f"# {questionnaire_description}"

    elif args.format == "latex":
        if args.query_prompt:
            query_str = args.query_prompt
        else:
            query_str = "Answer: ("

        questionnaire_description = \
            "\\documentclass{article}\n" + \
            "\\usepackage{enumitem}\n" + \
            "\n" + \
            "\\begin{document}\n" + \
            "\n" + \
            questionnaire_description

    elif args.format == "chat":
        if args.query_prompt:
            query_str = args.query_prompt
        else:
            query_str = "Answer: ("

    else:
        raise ValueError(f"Undefined format {args.format}.")


    # only no_profile with code formats change the questionnaire description
    assert (not questionnaire_description_empty == questionnaire_description) or (args.no_profile and args.format != "chat")

    prompt_skeleton = {
        "set_persona_str": set_persona_str,  # remove newline from the end
        "questionnaire_description": questionnaire_description,
        "query_str": f"{query_str}",
    }

    return prompt_skeleton


def dummy_lprobs_from_generation(response, answers, label_2_text_option_dict):


    def find_first_match(response, labels_strings, case_insensitive):

        if case_insensitive:
            labels_strings = [(l, s.lower()) for l, s in labels_strings]
            response = response.lower()

        for l, s in labels_strings:
            if s in response:
                return l, s

        return None, None

    # first try to match substrings
    # sort from longest to shortest (to avoid substrings, "Like me" vs "A little like me")
    labels_text_options = sorted(label_2_text_option_dict.items(), key=lambda x: len(x[1]), reverse=True)
    label, option = find_first_match(response, labels_text_options, case_insensitive=True)

    if option is not None:
        lprobs = [-0.01 if a == label else -100 for a in answers]
        return lprobs

    def find_matches(strings):
        lprobs = [-100] * len(strings)
        for i, op in enumerate(strings):
            if op in response:
                lprobs[i] = -0.01

        match = any([lp > -100 for lp in lprobs])

        return lprobs, match


    # look for 'A.'
    lprobs, match = find_matches([f"{a}." for a in answers])
    if match:
        return lprobs

    # look for "A "
    lprobs, _ = find_matches([f"{a} " for a in answers])
    if match:
        return lprobs

    # look for "A"
    lprobs, _ = find_matches(answers)
    return lprobs


def softmax(x):
    z = x - max(x)
    numerator = np.exp(z)
    denominator = np.sum(numerator)
    softmax = numerator/denominator
    return softmax


def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s


def format_example(df, idx, subject, experiment_name, args, permutations_dict, simulated_participant, include_answer=True):
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

    if args.format == "chat":
        for ch in choices[:num_options]:
            item_str += "\n({}) {}".format(ch, options_strings[permutations_dict[ch]])

    elif args.format == "code_py":
        item_str += "\n\"\"\"\n\n"

        item_str += "# Define the answers dictionary\n"
        item_str += "answers_dict = {\n"

        for ch in choices[:num_options]:

            item_str += "\t\"({})\": \"{}\",\n".format(ch, options_strings[permutations_dict[ch]])

        item_str += "}\n"

    elif args.format == "code_cpp":
        item_str += "\n)\";\n\n"

        item_str += "\t// Define the answers dictionary\n"
        item_str += "\tstd::map<std::string, std::string> answers_dict = {\n"

        for ch in choices[:num_options]:

            item_str += "\t\t{\"("+ch+")\", \""+options_strings[permutations_dict[ch]]+"\"},\n"

        item_str += "\t};\n"

    elif args.format == "conf_toml":

        item_str = item_str.replace("\n", "\n# ")
        item_str = f"# {item_str}"

        for ch in choices[:num_options]:
            item_str += f"\n# ({ch}) {options_strings[permutations_dict[ch]]}"

    elif args.format == "latex":
        item_str += "\n\\begin{enumerate}[label=(\\Alph*)]\n"

        for ch in choices[:num_options]:
            item_str += f"\t\\item {options_strings[permutations_dict[ch]]}\n"

        item_str += "\\end{enumerate}"

    else:
        raise ValueError(f"Undefined textual format {args.format}.")

    prompt = get_prompt_skeleton(
        subject=subject, experiment_name=experiment_name, args=args, simulated_participant=simulated_participant
    )

    prompt["item_str"] = item_str

    # query_in_reply will put query in the models response, if not add it to prompt here

    # if not args.query_in_reply:
    #     item_str += "\n"+prompt_skeleton["query"]

    if include_answer:
        prompt["answer"] = df.iloc[idx, k + 1]
        # item_str += " {}\n\n".format(df.iloc[idx, k + 1])

    # return prompt, num_options, prompt_skeleton

    return prompt, num_options


def hash_chat_conv(msgs_conv):
    json_string = json.dumps(msgs_conv)

    # Create a SHA256 hash of the string
    hash_object = hashlib.sha256(json_string.encode())

    # Get the hexadecimal representation of the hash
    hex_dig = hash_object.hexdigest()

    return hex_dig


def eval(args, subject, engine, dev_df, test_df, participant_perm_dicts, llm_generator=None, simulated_participant=None):
    cors = []
    all_probs = []
    all_lprobs = []
    all_answers = []
    all_generations = []
    all_scores = []

    # hashing for simulated conversations
    messages_conv = None
    messages_conv_hash = None

    gpt_token_counter = 0

    assert test_df.shape[0] == len(participant_perm_dicts)

    for item_i, permutations_dict in enumerate(participant_perm_dicts):
        inv_permutations_dict = {v: k for k, v in permutations_dict.items()}

        if item_i % 10 == 0:
            print(f"Eval progress: {item_i}/{test_df.shape[0]}")

        #  e.g. A -> A little like me
        label_2_text_option_dict = {
            label: test_df.iloc[item_i, score+1] for label, score in permutations_dict.items()
        }
        prompt, n_options = format_example(
            test_df, item_i,
            subject=subject,
            experiment_name=args.experiment_name,
            include_answer=False,
            args=args,
            permutations_dict=permutations_dict,
            simulated_participant=simulated_participant
        )

        skip_generation = False
        if "wvs_svas" in args.data_dir:

            if item_i == 6:
                assert "choose up to five" in prompt['item_str']
                previous_mentions = []

            elif 6 < item_i < 11:
                assert "choose up to five" in prompt['item_str']

                # other formats don't have the implementation for questions 6-16 (choose up to five)
                assert args.format == "chat"
                previous_mentions_labels = [map_number_to_choice(pm, inv_permutations_dict) for pm in previous_mentions]
                prompt["query_str"] = prompt['query_str'] + "), (".join(previous_mentions_labels + [''])

            elif item_i > 16:
                assert not "choose up to five" in prompt['item_str']

            if 10 < item_i < 17:
                skip_generation = True

        assert n_options == len(permutations_dict)
        answers = choices[:n_options]

        assert all([a in permutations_dict for a in answers])

        label = test_df.iloc[item_i, test_df.shape[1]-1]
        assert label in answers + ["undef"]

        if args.estimate_gpt_tokens and not skip_generation:
            encoder = tiktoken.encoding_for_model('gpt-3.5-turbo-0301')
            assert encoder == tiktoken.encoding_for_model('gpt-4-0314')
            gpt_token_counter += len(encoder.encode(" ".join(prompt.values())))

        if args.simulate_conversation_theme and not skip_generation:

            set_persona_str = prompt["set_persona_str"]
            if messages_conv is None:
                print("SIMULATING CONVERSATION")
                messages_conv, messages_conv_hash = simulate_conversation(
                    args=args,
                    engine=engine,
                    sim_engine=engine,
                    model_set_persona_string=set_persona_str,
                    simulated_participant=simulated_participant,
                    llm_generator=llm_generator
                )
            else:
                print("LOADING CACHED CONVERSATION")
                assert hash_chat_conv(messages_conv) == messages_conv_hash

        if skip_generation:
            print("Skipping WVS choose up to five")
            generation = random.choice([f"{c}" for c in answers])
            lprobs = dummy_lprobs_from_generation(generation, answers, label_2_text_option_dict)

        elif engine == "dummy":

            messages = construct_messages(
                prompt=prompt,
                system_message=args.system_message,
                messages_conv=messages_conv if args.simulate_conversation_theme else None,
                add_query_str=not args.query_in_reply,
            )

            if args.query_in_reply:
                messages += [{
                    "role": "assistant",
                    "content": prompt['query_str']
                }]

            formatted_prompt = apply_base_model_template(
                messages,
                simulated_participant=simulated_participant,
                simulated_population_type=args.simulated_population_type,
                add_generation_prompt=True,
                assistant_label=simulated_participant_to_name(
                    simulated_participant, args.simulated_population_type).upper(),
                user_label="USER",
                system_label="CONTEXT"
            )

            print(f"************************\nFORMATTED PROMPT:\n{formatted_prompt}\n******************")

            generation = random.choice([f"{c}" for c in answers])
            # if re.search("various changes", messages[-2]['content']):
            #     generation = messages[-2]['content'][messages[-2]['content'].index(") Go") - 1:][:1]
            # else:
            #     generation = random.choice([f"{c}" for c in answers])

            lprobs = dummy_lprobs_from_generation(generation, answers, label_2_text_option_dict)

        elif engine == "interactive":
            # ask the user to choose
            generation = input(f"{prompt}")

            lprobs = dummy_lprobs_from_generation(generation, answers, label_2_text_option_dict)

        elif engine in ["zephyr-7b-beta"] or "Mistral-7B" in engine or "Mixtral" in engine:

            tokenizer, model = llm_generator

            messages = construct_messages(
                prompt=prompt,
                system_message=args.system_message,
                messages_conv=messages_conv if args.simulate_conversation_theme else None,
                add_query_str=not args.query_in_reply,
            )

            if args.base_model_template:
                formatted_prompt = apply_base_model_template(
                    messages,
                    simulated_participant=simulated_participant,
                    simulated_population_type=args.simulated_population_type,
                    add_generation_prompt=True,
                    assistant_label=simulated_participant_to_name(simulated_participant, args.simulated_population_type).upper(),
                    user_label="USER",
                    system_label="CONTEXT"
                )

            else:
                formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            if args.query_in_reply:
                formatted_prompt += f"{prompt['query_str']}"

            print(f"************************\nFORMATTED PROMPT:\n{formatted_prompt}\n******************")

            inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

            output = model.generate(
                **inputs,
                max_new_tokens=1,
                # temperature=0.0001,
                do_sample=False,
                top_p=1.0,
                return_dict_in_generate=True,
                output_scores=True
            )
            option_scores, generation, lprobs = parse_hf_outputs(output=output, tokenizer=tokenizer, answers=answers)

        elif engine in [
            *[f"llama_2_{s}_chat" for s in ["7b", "13b", "70b"]],
            *[f"llama_2_{s}" for s in ["7b", "13b", "70b"]],
            *["phi-2", "phi-1.5", "phi-1"],
            *[f"Qwen-{s}" for s in ["72B", "14B", "7B"]],
            *[f"Qwen-{s}-Chat" for s in ["72B", "14B", "7B"]],
        ]:

            tokenizer, model = llm_generator

            messages = construct_messages(
                prompt=prompt,
                system_message=args.system_message,
                messages_conv=messages_conv if args.simulate_conversation_theme else None,
                add_query_str=not args.query_in_reply,
            )

            if args.base_model_template:
                formatted_prompt = apply_base_model_template(
                    messages,
                    simulated_participant=simulated_participant,
                    simulated_population_type=args.simulated_population_type,
                    add_generation_prompt=True,
                    assistant_label=simulated_participant_to_name(simulated_participant, args.simulated_population_type).upper(),
                    user_label="USER",
                    system_label="CONTEXT"
                )

            else:
                formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            if args.query_in_reply:

                formatted_prompt += f"{prompt['query_str']}"

            print(f"************************\nFORMATTED PROMPT:\n{formatted_prompt}\n******************")

            inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

            # token match
            output = model.generate(
                **inputs,
                max_new_tokens=1,
                # do_sample=True,
                # temperature=0.5,
                # top_p=0.9,
                # top_k=50,
                # num_beams=1,
                # repetition_penalty=1.2,
                return_dict_in_generate=True,
                output_scores=True
            )

            option_scores, generation, lprobs = parse_hf_outputs(output=output, tokenizer=tokenizer, answers=answers)

        elif engine in ["gpt-3.5-turbo-0301", "gpt-4-0314", "gpt-3.5-turbo-0613", "gpt-4-0613", "gpt-3.5-turbo-1106-preview", "gpt-4-1106-preview"]:
            if args.query_in_reply:
                raise ValueError("Can't use query_in_reply with gpt models.")

            if args.base_model_template:
                raise ValueError("base_model_template not supported for gpt models")

            while True:
                try:

                    messages = construct_messages(
                        prompt=prompt,
                        system_message=args.system_message,
                        messages_conv=messages_conv if args.simulate_conversation_theme else None,
                        add_query_str=True,
                    )

                    print_chat_messages(messages)
                    encoder = tiktoken.encoding_for_model(engine)

                    # get the encoding for each letter in choices
                    logit_bias = {encoder.encode(c)[0]: 100 for c in answers}

                    c = openai.ChatCompletion.create(
                        model=engine,
                        messages=messages,
                        max_tokens=1,
                        n=1,
                        temperature=0,
                        logit_bias=logit_bias,
                        request_timeout=30,
                    )
                    generation = c['choices'][0]['message']['content']

                    break

                except Exception as e:
                    print(e)
                    print("Pausing")
                    time.sleep(10)
                    continue

            lprobs = dummy_lprobs_from_generation(generation, answers, label_2_text_option_dict)

        elif engine in ["gpt-3.5-turbo-instruct-0914"]:

            if args.simulate_conversation_theme:
                raise NotImplementedError("noisy conversation not implemented for user message")

            if args.system_message:
                raise NotImplementedError("Text generation models don't have system message")

            while True:
                try:
                    c = openai.Completion.create(
                        engine=engine,
                        prompt=prompt,
                        max_tokens=1,
                        logprobs=1,
                        temperature=0,
                        echo=True
                    )
                    break
                except Exception as e:
                    print(e)
                    print("Pausing")
                    time.sleep(1)
                    continue

            assert c["choices"][0]["logprobs"]["top_logprobs"][0] is None

            output_dict = dict([tuple(*i.items()) for i in c["choices"][0]["logprobs"]["top_logprobs"][1:]])
            option_scores = {ans: output_dict.get(ans, -100) for ans in answers}

            # take the most probable answer as the generation
            generation = max(option_scores, key=option_scores.get)

            # extract logprobs
            lprobs = [float(option_scores[a]) for a in answers]

        else:
            raise ValueError(f"Not recognized model {engine}.")

        probs = softmax(np.array(lprobs))
        pred = {i: c for i, c in enumerate(answers)}[np.argmax(lprobs)]
        cor = pred == label
        score = map_choice_to_number(pred, permutations_dict)

        print(colored(f"Pred:{pred} (Generation:{generation}; Score: {score})", "green"))
        print("------------------")

        if "wvs_svas" in args.data_dir:

            if 5 < item_i < 17:
                assert "choose up to five" in prompt['item_str']
            else:
                assert "choose up to five" not in prompt['item_str']

            if 5 < item_i < 11:
                assert not skip_generation
                previous_mentions.append(score)

        cors.append(cor)
        all_lprobs.append(lprobs)
        all_probs.append(probs)
        all_answers.append(pred)
        all_generations.append(generation)
        all_scores.append(score)

        if "wvs_svas" in args.data_dir and item_i == 16:
            assert len(all_scores) == 17

            assert skip_generation
            # parse previous answers and put "mentioned/not mentioned"

            assert len(previous_mentions) == 5
            previous_mentions = list(set(previous_mentions))

            for op_i in range(n_options):
                # mentioned -> 1
                # not mentioned -> 2
                mention_score = 1 if (op_i+1 in previous_mentions) else 2
                mention_generation = answers[0] if mention_score == 1 else answers[1]
                mention_answers = answers[:2]
                mention_label_2_text_option_dict = dict(zip(mention_answers, ["Mention", "Don't mention"]))
                mention_lprobs = dummy_lprobs_from_generation(
                    mention_generation, mention_answers, mention_label_2_text_option_dict
                )
                mention_probs = softmax(np.array(mention_lprobs))
                mention_pred = {i: c for i, c in enumerate(mention_answers)}[np.argmax(mention_lprobs)]
                mention_cor = False

                # lprobs, probs, pred, generation, score
                mention_index = 6 + op_i
                cors[mention_index] = mention_cor
                all_lprobs[mention_index] = mention_lprobs
                all_probs[mention_index] = mention_probs
                all_answers[mention_index] = mention_pred
                all_generations[mention_index] = mention_generation
                all_scores[mention_index] = mention_score

    acc = np.mean(cors)
    cors = np.array(cors)
    all_scores = np.array(all_scores)

    print("Average accuracy {:.3f} - {}".format(acc, subject))

    if args.estimate_gpt_tokens:
        print("total GPT tokens used: {} for subject {}".format(gpt_token_counter, subject))
        print(f"\tgpt-4 ~ {0.04 *gpt_token_counter/1000:.4f} dollars".format(args))
        print(f"\tgpt-3.5 ~ {0.002 *gpt_token_counter/1000:.4f} dollars".format(args))
        print(f"\tdavinci ~ {0.02 *gpt_token_counter/1000:.4f} dollars".format(args))
        print(f"\tcurie ~ {0.002 *gpt_token_counter/1000:.4f} dollars".format(args))
        print(f"\tbabagge ~ {0.0005 *gpt_token_counter/1000:.4f} dollars".format(args))
        print(f"\tada ~ {0.0004 *gpt_token_counter/1000:.4f} dollars".format(args))

    return cors, acc, all_probs, all_lprobs, all_answers, all_scores, all_generations, gpt_token_counter

def remove_prefix(s, pref):
    if s.startswith(pref):
        return s[len(pref):]
    return s


def main(args):
    engine = args.engine
    print("Engine:", engine)

    subjects = sorted([f.split("_test.csv")[0] for f in os.listdir(os.path.join(args.data_dir, "test")) if "_test.csv" in f])

    # dump results dir
    dump_results_dir = os.path.join(args.save_dir, "_".join([
        args.experiment_name,
        engine,
        os.path.basename(args.data_dir),
        f"{args.pvq_version}_" if args.pvq_version else "",
        f"permutations_{args.permutations}" if args.permutations > 1 else "",
        f"permute_options_{args.permute_options_seed}" if args.permute_options else "",
        f"no_profile_{args.no_profile}" if args.no_profile else "",
        f"format_{args.format}",
        f"weather_{args.weather}" if args.weather else "",
        f"simulate_conv_{args.simulate_conversation_theme}" if args.simulate_conversation_theme else "",
        timestamp
    ]))
    os.makedirs(dump_results_dir, exist_ok=True)
    print("Savedir: ", dump_results_dir)

    # Data preparation
    if len(subjects) == 0:
        raise ValueError("No subjects found.")

    if "data_pvq" in args.data_dir:
        assert "pvq" in args.experiment_name

        subjects_to_evaluate = [args.pvq_version]
        if args.pvq_version not in ["pvq_female", "pvq_male", "pvq_auto"]:
            raise ValueError(f"Unknown pvq version {args.pvq_version}.")

        # assert set(subjects_to_evaluate).issubset(subjects)
        subjects = subjects_to_evaluate

    print("Args:", args)
    print("Subjects:", subjects)

    gpt_tokens_total = 0

    if engine in ["zephyr-7b-beta"]:
        print("Loading zephyr-7b-beta")
        tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta", cache_dir=hf_cache_dir, device_map="auto")
        model = AutoModelForCausalLM.from_pretrained("HuggingFaceH4/zephyr-7b-beta", torch_dtype=torch.bfloat16, device_map="auto", cache_dir=hf_cache_dir)

        llm_generator = (tokenizer, model)

    elif engine in [
        *[f"Qwen-{s}" for s in ["72B", "14B", "7B"]],
        *[f"Qwen-{s}-Chat" for s in ["72B", "14B", "7B"]]
    ]:
        tokenizer = AutoTokenizer.from_pretrained(f"Qwen/{engine}", trust_remote_code=True, cache_dir=hf_cache_dir)
        model = AutoModelForCausalLM.from_pretrained(f"Qwen/{engine}", device_map="auto", trust_remote_code=True, cache_dir=hf_cache_dir).eval()
        llm_generator = (tokenizer, model)

    elif engine in ["phi-1", "phi-1.5", "phi-2"]:
        # Load model directly
        # torch.set_default_device("cuda")
        # model = AutoModelForCausalLM.from_pretrained(f"microsoft/{engine}", torch_dtype="auto", trust_remote_code=True, cache_dir=hf_cache_dir)
        tokenizer = AutoTokenizer.from_pretrained(f"microsoft/{engine}", trust_remote_code=True, cache_dir=hf_cache_dir)
        model = AutoModelForCausalLM.from_pretrained(f"microsoft/{engine}", trust_remote_code=True, cache_dir=hf_cache_dir, device_map="cuda")
        llm_generator = (tokenizer, model)

    elif engine in [
        "falcon-7b", "falcon-7b-instruct",
        "falcon-40b", "falcon-40b-instruct",
        "falcon-180b", "falcon-180b-chat",
    ]:

        # loads on SWAP for some reason ->problem with device map?
        tokenizer = AutoTokenizer.from_pretrained(f"tiiuae/{engine}", cache_dir=hf_cache_dir, device_map="cuda")

        pipe = pipeline(
            "text-generation",
            model=f"tiiuae/{engine}",
            tokenizer=tokenizer,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="cuda",
            cache_dir=hf_cache_dir,
        )
        # path='/gpfsscratch/rech/imi/utu57ed/.cache/huggingface/models--tiiuae--falcon-7b'
        path='/gpfsscratch/rech/imi/utu57ed/.cache/huggingface/models--tiiuae--falcon-7b/snapshots/898df1396f35e447d5fe44e0a3ccaaaa69f30d36'
        model = AutoModelForCausalLM.from_pretrained(
            path, device_map="cuda", torch_dtype=torch.float16, trust_remote_code=True
        )

        model = AutoModelForCausalLM.from_pretrained(
            f"tiiuae/{engine}",
            # torch_dtype=torch.bfloat16,
            cache_dir=hf_cache_dir,
            trust_remote_code=True,
            # device_map="cuda",
        )

    elif engine in [
        "Mistral-7B-v0.1",
        "Mistral-7B-Instruct-v0.2",
        "Mistral-7B-Instruct-v0.1",
    ]:
        print(f"Loading {engine}")
        tokenizer = AutoTokenizer.from_pretrained(f"mistralai/{engine}", cache_dir=hf_cache_dir, device_map="auto")
        model = AutoModelForCausalLM.from_pretrained(f"mistralai/{engine}", device_map="auto", cache_dir=hf_cache_dir)

        llm_generator = (tokenizer, model)
        
    elif engine in [
        "Mixtral-8x7B-v0.1",
        "Mixtral-8x7B-Instruct-v0.1",
    ]:
        print(f"Loading {engine}")
        tokenizer = AutoTokenizer.from_pretrained(f"mistralai/{engine}", cache_dir=hf_cache_dir, device_map="auto")
        model = AutoModelForCausalLM.from_pretrained(
            f"mistralai/{engine}", device_map="auto", cache_dir=hf_cache_dir, torch_dtype=torch.float16, attn_implementation="flash_attention_2")

        llm_generator = (tokenizer, model)

    elif engine in [
        "Mixtral-8x7B-v0.1-4b",
        "Mixtral-8x7B-Instruct-v0.1-4b",
    ]:
        model_name = engine.rstrip("-4b")
        print(f"Loading {engine} -> {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(f"mistralai/{model_name}", cache_dir=hf_cache_dir, device_map="auto")
        model = AutoModelForCausalLM.from_pretrained(f"mistralai/{model_name}", device_map="auto", cache_dir=hf_cache_dir, load_in_4bit=True, attn_implementation="flash_attention_2")

        llm_generator = (tokenizer, model)

    elif engine in [
        *[f"llama_2_{s}_chat" for s in ["7b", "13b", "70b"]],
        *[f"llama_2_{s}" for s in ["7b", "13b", "70b"]],
    ]:

        print("Loading llama 2")
        import re

        model_size = re.findall(r"_(\d+b)", engine)[0]
        chat = "chat" in engine

        hf_model_name = f"meta-llama/Llama-2-{model_size}-{'chat-' if chat else ''}hf"

        tokenizer = AutoTokenizer.from_pretrained(hf_model_name, token=hf_token, cache_dir=hf_cache_dir)

        if not chat:
            # monkey patch
            hf_model_name_chat = f"meta-llama/Llama-2-{model_size}-chat-hf"
            tokenizer_chat = AutoTokenizer.from_pretrained(hf_model_name_chat, token=hf_token, cache_dir=hf_cache_dir)
            tokenizer.apply_chat_template = tokenizer_chat.apply_chat_template

        model = AutoModelForCausalLM.from_pretrained(hf_model_name, torch_dtype=torch.float16, token=hf_token, device_map="auto", cache_dir=hf_cache_dir)

        llm_generator = (tokenizer, model)

    elif "gpt" in engine or engine in ["dummy", "interactive"]:
        llm_generator = None

    else:
        raise ValueError(f"Undefined model: {engine}")

    print(f"Loaded model: {args.engine}.")

    if "hofstede" in args.data_dir:
        max_n_options = 5
    elif "wvs_svas" in args.data_dir:
        max_n_options = 15
    elif "pvq" in args.data_dir:
        max_n_options = 6
    elif "big5" in args.data_dir:
        max_n_options = 5
    elif "donation" in args.data_dir:
        max_n_options = 6
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

    # elif args.simulated_population_type == "anes":
    #     # https://electionstudies.org/
    #     with open("personas/ANES/voters2016_50.json") as f:
    #         simulated_population = json.load(f)
    #
    # elif args.simulated_population_type in ["llm_personas", "user_personas"]:
    #     with open("personas/personachat/grammar/personachat_I.txt") as f:
    #         simulated_population = [bio.rstrip() for bio in f.readlines()]

    all_cors = []

    # list because of permutations
    subj_acc = [{} for _ in range(len(simulated_population))]
    subj_lprobs = [{} for _ in range(len(simulated_population))]
    subj_len = [{} for _ in range(len(simulated_population))]
    metrics = [{} for _ in range(len(simulated_population))]
    answers = [{} for _ in range(len(simulated_population))]
    generations = [{} for _ in range(len(simulated_population))]

    # evaluate model
    for subject in subjects:

        dev_df = None
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

            if "wvs_svas" in args.data_dir:
                # wvs has a varying number of options
                test_df = pd.read_csv(
                    os.path.join(args.data_dir, args.eval_set, subject + f"_{args.eval_set}.csv"),
                    header=None,
                    names=range(max_n_options),
                    # keep_default_na=False,
                )
                test_df.fillna("undef", inplace=True)

                max_n_options_list = np.array([max_n_options] * len(test_df))
                undef_counts = np.array(test_df.apply(lambda row: (row == 'undef').sum(), axis=1))
                n_options = max_n_options_list - undef_counts - 1


            else:
                test_df = pd.read_csv(
                    os.path.join(args.data_dir, args.eval_set, subject + f"_{args.eval_set}.csv"),
                    header=None,
                    keep_default_na=False,
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

        # evaluate over population
        for sim_part_i, (simulated_participant, simulated_participant_gender, participant_perm_dicts) in enumerate(zip(simulated_population, simulated_population_genders, permutations_dicts)):
            print(f"Simulated participant {sim_part_i}")

            if subject == "pvq_auto":
                test_df = test_df_dict[simulated_participant_gender]

            cors, acc, eval_probs, eval_lprobs, preds, preds_values, gens, gpt_tokens = eval(
                args=args,
                subject=subject,
                engine=engine,
                dev_df=dev_df,
                test_df=test_df,
                participant_perm_dicts=participant_perm_dicts,
                llm_generator=llm_generator,
                simulated_participant=simulated_participant,
            )
            all_cors.append(cors)
            gpt_tokens_total += gpt_tokens

            subj_acc[sim_part_i][subject] = acc
            subj_lprobs[sim_part_i][subject] = eval_lprobs
            subj_len[sim_part_i][subject] = num_questions
            answers[sim_part_i][subject] = list(zip(preds, map(int, preds_values)))
            generations[sim_part_i][subject] = gens

            if "hofstede" in args.data_dir:
                assert "hofstede" in args.experiment_name

                # from the manual (question indices start from 1)
                # power_distance = 35(m07 – m02) + 25(m20 – m23) + C(pd)
                # individualism = 35(m04 – m01) + 35(m09 – m06) + C(ic)
                # masculinity = 35(m05 – m03) + 35(m08 – m10) + C(mf)
                # uncertainty_avoidance = 40(m18 - m15) + 25(m21 – m24) + C(ua)
                # long_term_orientation = 40(m13 – m14) + 25(m19 – m22) + C(ls)
                # indulgence = 35(m12 – m11) + 40(m17 – m16) + C(ir)

                # indices start from 0
                metrics[sim_part_i][subject] = {
                    "Power Distance": 35*(preds_values[6] - preds_values[1]) + 25*(preds_values[19] - preds_values[22]),
                    "Individualism": 35*(preds_values[3] - preds_values[0]) + 35*(preds_values[8] - preds_values[5]),
                    "Masculinity": 35*(preds_values[4] - preds_values[2]) + 35*(preds_values[7] - preds_values[9]),
                    "Uncertainty Avoidance": 40*(preds_values[17] - preds_values[14]) + 25*(preds_values[20] - preds_values[23]),
                    "Long-Term Orientation": 40*(preds_values[12] - preds_values[13]) + 25*(preds_values[18] - preds_values[21]),
                    "Indulgence": 35*(preds_values[11] - preds_values[10]) + 40*(preds_values[16] - preds_values[15])
                }

                metrics[sim_part_i][subject] = {k: float(v) for k, v in metrics[sim_part_i][subject].items()}

            elif "wvs_svas" in args.data_dir:
                # per participant metrics don't exists
                metrics[sim_part_i][subject] = {'answers': list(map(int, preds_values))}

            elif "big5" in args.data_dir:

                # items are given in the following order
                # positive items for Neuroticism, neg items for Neuroticism,
                # positive items for Extravesion, neg items for Extraversion,
                # ...
                # positive items for Conscientiousness, neg items for Conscientiousness

                if "data_big5_50" == args.data_dir:
                    items_per_chunk = 5
                elif "data_big5_100" == args.data_dir:
                    items_per_chunk = 10
                else:
                    raise ValueError(f"data_dir {args.data_dir} not supported.")

                # separate answers into chunks
                chunks = [preds_values[st:st + items_per_chunk] for st in range(0, len(preds_values), items_per_chunk)]

                # pos item score - A = 1, F = 5
                # neg item score - A = 5, F = 1
                # revert negative items - neg_i = 6 - neg_i
                # i.e. total: sum(pos_its) + 6*items_per_chunk - sum(nef_its)
                metrics[sim_part_i][subject] = {
                    "Neuroticism": chunks[0].sum() + 6*items_per_chunk - chunks[1].sum(),
                    "Extraversion": chunks[2].sum() + 6*items_per_chunk - chunks[3].sum(),
                    "Openness to Experience": chunks[4].sum() + 6*items_per_chunk - chunks[5].sum(),
                    "Agreeableness": chunks[6].sum() + 6*items_per_chunk - chunks[7].sum(),
                    "Conscientiousness": chunks[8].sum() + 6*items_per_chunk - chunks[9].sum()
                }
                metrics[sim_part_i][subject] = {k: float(v) for k, v in metrics[sim_part_i][subject].items()}

            elif "pvq" in args.data_dir:
                assert "pvq" in args.experiment_name

                profile_values_idx_json = os.path.join(os.path.join(args.data_dir, "raw"), "values.json")

                with open(profile_values_idx_json) as f:
                    profile_values_idx = json.load(f)

                profile_values_idx = {k: np.array(v)-1 for k, v in profile_values_idx.items() if k != "_comment"}

                metrics[sim_part_i][subject] = {}

                for profile_value, idxs in profile_values_idx.items():
                    metrics[sim_part_i][subject][profile_value] = preds_values[idxs].mean() # legacy: todo: remove and save those below

            elif "donation" in args.data_dir:
                assert "donation" in args.experiment_name

                if "tolkien_donation" in args.data_dir:

                    groups = ["elves", "dwarves", "orcs", "humans", "hobbits"]

                    donated = (preds_values-1)*2
                    group_donations = np.split(donated, len(groups))
                    assert set([len(g) for g in group_donations]) == {20}

                    metrics[sim_part_i][subject] = {
                        f"Donation {g}": np.mean(g_d) for g, g_d in zip(groups, group_donations)
                    }

                else:
                    raise ValueError("Unknown donation type: {args.data_dir}.")

            elif "ultimatum" in args.data_dir:
                assert "ultimatum" in args.experiment_name

                if "tolkien_ultimatum" in args.data_dir:

                    groups = [
                        "elves_fair",
                        "elves_unfair",
                        "dwarves_fair",
                        "dwarves_unfair",
                        "orcs_fair",
                        "orcs_unfair",
                        "humans_fair",
                        "humans_unfair",
                        "hobbits_fair",
                        "hobbits_unfair",
                    ]

                    accepted = preds_values == 1
                    group_acceptances = np.split(accepted, len(groups))
                    assert len(group_acceptances[0]) == 10  # 8 x 10 questions

                    # 1 - Accept; 2 - Reject;
                    metrics[sim_part_i][subject] = {
                        f"Acceptance Rate {g}": np.mean(g_a) for g, g_a in zip(groups, group_acceptances)
                    }

                elif "regular_ultimatum":
                    # accepted = preds_values == 1
                    # metrics[sim_part_i][subject] = {
                    #     f"Acceptance Rate": np.mean(accepted)
                    # }

                    groups = [
                        "baa_fair", "baa_unfair",
                        "w_fair", "w_unfair",
                        "anhpi_fair", "anhpi_unfair",
                        "aian_fair", "aian_unfair",
                        "hl_fair", "hl_unfair",
                    ]

                    accepted = preds_values == 1
                    group_acceptances = np.split(accepted, len(groups))
                    assert len(group_acceptances[0]) == 10  # 10 x 10 questions

                    # 1 - Accept; 2 - Reject;
                    metrics[sim_part_i][subject] = {
                        f"Acceptance Rate {g}": np.mean(g_a) for g, g_a in zip(groups, group_acceptances)
                    }

                else:
                    raise ValueError("undefined ultimatum experiment")

            elif "tolkien_good_bad":
                accepted = preds_values == 1
                metrics[sim_part_i][subject] = {
                    f"Protagonist score": np.mean(accepted)
                }

            elif "tolkien_public_goods" in args.data_dir:
                assert len(preds_values) == 4
                metrics[sim_part_i][subject] = {
                    g: float(preds_values[i]-1)*10 for i, g in enumerate(["elves", "dwarves", "orcs", "humans"])
                }

            else:
                raise NotImplementedError("Evaluation not implemented")

            # res_test_df = test_df.copy()
            # res_test_df["{}_correct".format(engine)] = cors
            # for j in range(eval_probs.shape[1]):
            #     choice = choices[j]
            #     res_test_df["{}_choice{}_probs".format(engine, choice)] = eval_probs[:, j]

        # aggregate to means
        mean_subj_acc = defaultdict(list)
        for subj_acc_perm in subj_acc:
            for k, v in subj_acc_perm.items():
                mean_subj_acc[k].append(v)
        mean_subj_acc = {k: np.mean(v) for k,v in mean_subj_acc.items()}

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

        weighted_acc = np.mean(np.concatenate(all_cors))

        pop_metrics = {}

        if "wvs_svas" in args.data_dir:

            if args.simulated_population_type == "tolkien_characters":

                with open("personas/tolkien_characters/tolkien_characters_races.txt", "r") as file:
                    races = [line.rstrip('\n') for line in file]

                elves_inds = np.where(np.array(races) == "el")[0]
                humans_inds = np.where(np.array(races) == "hu")[0]
                bg_inds = np.where(np.array(races) == "bg")[0]

                tolk_characters_races_inds = {
                    "all": list(range(len(simulated_population))),
                    "elves": elves_inds,
                    "humans": humans_inds,
                    "bad guys": bg_inds,
                }

                for race, inds in tolk_characters_races_inds.items():
                    race_answers = np.array([metrics[i]['wvs_svas']['answers'] for i in inds])
                    pop_metrics[race] = {"hist": [], "distr": []}

                    for it_i, n_opt in enumerate(n_options):
                        hist = dict(zip(*np.unique(race_answers[:, it_i], return_counts=True)))
                        hist = {int(k): int(v) for k,v in hist.items()}
                        pop_metrics[race]["hist"].append(hist)

                        # # for up to five questions - mention, don't mention
                        # if 5 < it_i < 17:
                        #     n_opt = 2

                        N = sum(hist.values())
                        distr = {opt: n/N for opt, n in hist.items()}
                        assert len(distr) >= len(hist)
                        pop_metrics[race]["distr"].append(distr)

            else:
                raise NotImplementedError("Other populations are not implemented for wvs_svas")

        # save results
        for subj, m in mean_metrics.items():
            if m:
                print("Subject: ", subj)
                for metric, score in m.items():
                    print(f"{metric} : {score}")

                plot_dict(m, savefile=os.path.join(dump_results_dir, f"plot_{subj}.png"))

        # accuracy
        plot_dict(mean_subj_acc, savefile=os.path.join(dump_results_dir, f"plot_mean_acc.png"))

        if not os.path.exists(dump_results_dir):
            os.mkdir(dump_results_dir)

        json_dump_path = os.path.join(dump_results_dir, 'results.json')

        with open(json_dump_path, 'w') as fp:
            json.dump({
                "args": vars(args),
                **mean_subj_acc,
                **{
                    "average": weighted_acc
                },
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

        print("")
        print("Average accuracy per subject.")
        for subject in subjects:
            print("{} accuracy ({}): {:.3f}".format(subject, subj_len[subject], mean_subj_acc[subject]))

        print("Average accuracy: {:.3f}".format(weighted_acc))

        if pop_metrics:
            print("pop metrics:", pop_metrics['all']['hist'])

    if args.estimate_gpt_tokens:
        print("total GPT tokens used: {}".format(gpt_tokens_total))
        print(f"\tgpt-4 ~ {0.04 *gpt_tokens_total/1000:.4f} dollars".format(args))
        print(f"\tgpt-3.5 ~ {0.002 *gpt_tokens_total/1000:.4f} dollars".format(args))
        print(f"\tdavinci ~ {0.02 *gpt_tokens_total/1000:.4f} dollars".format(args))
        print(f"\tcurie ~ {0.002 *gpt_tokens_total/1000:.4f} dollars".format(args))
        print(f"\tbabagge ~ {0.0005 *gpt_tokens_total/1000:.4f} dollars".format(args))
        print(f"\tada ~ {0.0004 *gpt_tokens_total/1000:.4f} dollars".format(args))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", "-d", type=str, required=True)
    parser.add_argument("--save_dir", "-s", type=str, default="results/results_test")
    parser.add_argument("--experiment_name", "-n", type=str, default="")
    parser.add_argument("--pvq-version", type=str, default="pvq_auto", choices=["pvq_female", "pvq_male", "pvq_auto"])
    parser.add_argument("--engine", "-e", type=str, default="dummy")
    parser.add_argument("--format", type=str, default="chat", choices=["chat", "code_py", "code_cpp", "conf_toml", "latex"])
    parser.add_argument("--weather", type=str, default=None)
    parser.add_argument('--profile', type=str, help='Profile definition in format "k:v;k:v;k:v", ex. "age:35;interests:reading books"')
    parser.add_argument("--max-tokens", "-mt", type=int, default=10, help="How many tokens to generate in genreative-qa")
    parser.add_argument("--query-in-reply", action="store_true", help="Force the query string as the beginning of the model's reply.")
    parser.add_argument("--base-model-template", action="store_true")
    parser.add_argument("--query-prompt", "-qp", type=str, help="Use Answer(as ONE letter): where applicable.")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--system-message", "-sm", action="store_true")
    parser.add_argument("--assert-params", action="store_true")
    parser.add_argument("--direct-perspective", action="store_true")
    parser.add_argument("--cold-run", "-cr", action="store_true")
    parser.add_argument("--estimate-gpt-tokens", "-t", action="store_true")
    parser.add_argument("--eval-set", type=str, default="test", choices=["test", "val"])
    parser.add_argument("--no-profile", action="store_true")
    parser.add_argument("--simulate-conversation-theme", type=str, default=None)
    parser.add_argument("--simulated-conversation-n-messages", type=int, default=5)
    parser.add_argument("--permute-options", "-po", action="store_true")
    parser.add_argument("--simulated-human-knows-persona", action="store_true")
    parser.add_argument("--simulated-population-type", "-pop", type=str, default="tolkien_characters", choices=["permutations", "tolkien_characters", "famous_people", "llm_personas", "user_personas", "anes"])
    parser.add_argument("--permutations", "-p", type=int, default=1)  # permutations as a population type
    parser.add_argument("--permute-options-seed", type=str)
    parser.add_argument("--separator", action="store_true")
    args = parser.parse_args()

    if not args.data_dir.startswith("data"):
        raise ValueError(f"data_dir should be inside data, and it's {args.data_dir}")

    if args.assert_params:
        # check parameters for models
        if "gpt" in args.engine and "instruct" not in args.engine:
            assert args.system_message
            assert not args.query_in_reply

        if args.engine in ["phi-2", "phi-1.5", "phi-1", "Qwen-72B", "Qwen-14B", "Qwen-7B"]:
            # phi is a base model
            assert args.query_in_reply
            assert args.system_message
            assert args.base_model_template

        if ("llama_2" in args.engine and "chat" in args.engine) or "zephyr" in args.engine:
            assert args.system_message
            assert args.query_in_reply
            assert not args.base_model_template

        if "llama_2" in args.engine and "chat" not in args.engine:
            # base llama_2 model
            assert args.system_message
            assert args.query_in_reply
            assert args.base_model_template

        if "Mistral" in args.engine or "Mixtral" in args.engine:
            assert args.query_in_reply

            if "Instruct" in args.engine:
                assert not args.system_message
                assert not args.base_model_template
            else:
                # base model
                assert args.system_message
                assert args.base_model_template

    if args.base_model_template:
        if not args.system_message:
            raise ValueError("Use system-message with base_model_template -> system is parsed to 'CONTEXT:' ")

    if args.simulated_population_type == "permutations":
        if args.simulated_human_knows_persona:
            raise ValueError("Use simulated_human_knows_persona cannot be used with permutations sim. population type")
    else:
        if args.simulate_conversation_theme and not args.simulated_human_knows_persona:
            raise ValueError("Use simulated_human_knows_persona.")


    if args.simulate_conversation_theme in ["None", "none"]:
        args.simulate_conversation_theme = None

    if args.estimate_gpt_tokens:
        if "gpt" not in args.engine and args.engine != "dummy":
            raise ValueError("Only gpt-4 gpt-3 and dummy support estimating GPT tokens")

    if args.permute_options and args.permute_options_seed is None:
        raise ValueError("Permute options string should be defined for stability")

    if args.cold_run:
        print("2nd person:", args.direct_perspective)
        print("System message:", args.system_message)
        # just used to show the profile to be used
        exit()

    if not args.separator and not args.no_profile:
        raise ValueError("You are not using a separator?")

    if ("gpt-3.5" in args.engine and args.permutations > 50) or ("gpt-4" in args.engine and args.permutations > 5):
        raise ValueError(f"Are you sure you want to use {args.permutations} with {args.engine}??")

    # assert for plosone gpt or query_in_reply for other models
    # because query_in_reply can't be implemented for GPTs
    assert args.query_in_reply or "gpt" in args.engine

    if "gpt" in args.engine:
        if args.query_in_reply:
            raise ValueError("Can't use query in reply with gpt models")

    assert args.no_profile

    if "pvq" in args.data_dir:
        assert args.pvq_version == "pvq_auto"

    start_time = time.time()
    main(args)
    end_time = time.time()
    print("Elapsed time:", str(datetime.timedelta(seconds=end_time-start_time)).split(".")[0])


