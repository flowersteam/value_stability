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
import sys

from termcolor import colored

hostname = os.uname()[1]
if hostname == "PTB-09003439":
    hf_cache_dir = "/home/flowers-user/.cache/huggingface"
elif "plafrim" in hostname:
    hf_cache_dir ="/beegfs/gkovac/hf_cache_dir"
else:
    hf_cache_dir = "/gpfsscratch/rech/imi/utu57ed/.cache/huggingface"

os.environ['TRANSFORMERS_CACHE'] = hf_cache_dir
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList, TextStreamer
import transformers

import json
import hashlib

def apply_chat_template_for_base_model(messages, add_generation_prompt=True):
    formatted_conversation = ""

    for msg in messages:
        formatted_conversation += f"{msg['role'].upper()}:{msg['content']}"
        formatted_conversation += "\n-----------------------------------\n"

    if add_generation_prompt:
        formatted_conversation += f"ASSISTANT:"

    return formatted_conversation

# take the theme starter
opening_questions_for_themes = {
    "poem": "Hello, let's write a poem together. You start by the first verse I'll add the second one, and so on.",
    "joke": "Tell me a joke.",
    "history": "What is the significance of the battle of Hastings. Answer in two sentences.",  # slight collapse
    "chess": "1. e4",
    "grammar": "Can you check this sentence for grammar? \n Whilst Jane was waiting to meet hers child their nose started bleeding.",
    "religion": "Why do we believe in the assumption of Mary? Please answer in two sentences.",
    "tax": "Can you explain how can my company pay less tax in two sentences?",
    "vacation": "I was thinking of going surfing is that relaxing? Please reply briefly.",
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

def simulate_conversation(args, engine, sim_engine, model_system_message=None, llm_generator=None):
    # only simulate the conversation once, use the same in all permutations
    if llm_generator is not None:
        tokenizer, model = llm_generator

    opening_question = opening_questions_for_themes[args.simulate_conversation_theme]

    conversation = [opening_question]

    # simulate conversation
    n_msgs = 5
    assert n_msgs % 2 == 1  # must be odd so that the first one is user role

    for msg_i in range(n_msgs):

        # assign roles to messages - alternating, last one user
        simulated_conv_messages = create_simulated_messages(conversation, last="user")

        if msg_i % 2 == 0:
            # even -> gpt as gpt
            assert simulated_conv_messages[0]['role'] == "user"

            if model_system_message:
                simulated_conv_messages = [{
                    "role": "system" if args.system_message else "user",
                    "content": model_system_message
                }] + simulated_conv_messages

            engine_ = engine

        else:
            # gpt as human
            assert simulated_conv_messages[0]['role'] == "assistant"
            simulated_conv_messages = [
                {"role": "system", "content": f"You are simulating a human using a chatbot."}
            ] + simulated_conv_messages

            engine_ = sim_engine

        if "gpt" in engine_:
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
            if simulated_conv_messages[1]['role'] == "assistant":
                # llama conversations cannot start with an assistant's message, insert dummy user message
                simulated_conv_messages.insert(1, {'role': 'user', 'content': ''})

            input_ids = tokenizer.apply_chat_template(simulated_conv_messages, return_tensors="pt", add_generation_prompt=True).to(model.device)
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
                output_scores=True
            )
            response = tokenizer.decode(output_seq.sequences[0][len(input_ids[0]):], skip_special_tokens=True)

        elif "zephyr" in engine_:
            input_ids = tokenizer.apply_chat_template(simulated_conv_messages, return_tensors="pt", add_generation_prompt=True).to(model.device)
            output_seq = model.generate(
                input_ids=input_ids,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                top_k=50,
                return_dict_in_generate=True,
                output_scores=True
            )
            response = tokenizer.decode(output_seq.sequences[0][len(input_ids[0]):], skip_special_tokens=True)

        elif "mistral" in engine_:
            raise NotImplementedError("Not implemented mistral sim conv.")

        else:
            raise NotImplementedError(f"Simulated conversations not implemented for {engine_}")

        conversation.append(response)

        messages_conv = create_simulated_messages(conversation, last="assistant")
        messages_conv_hash = hash_chat_conv(messages_conv)

    print_chat_messages(messages_conv)

    return messages_conv, messages_conv_hash


def map_choice_to_number(letter, permutations_dict):
    # A-F -> 1-6
    # find index of letter in choices and add 1

    number = permutations_dict[letter] + 1
    # assert number == choices.index(letter) + 1
    return number


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
import time
import sys

from crop import crop

openai.api_key = os.environ["OPENAI_API_KEY"]
hidden_key = openai.api_key[:10] + "*" * 10 + openai.api_key[20:]
print("OPENAI KEY:", hidden_key)

hf_token = os.environ["HF_TOKEN"]
hidden_token = hf_token[:6] + "*" * (len(hf_token)-12) + hf_token[-6:]
print("HF TOKEN:", hidden_token )


choices = ["A", "B", "C", "D", "E", "F"]
# choices = ["a", "b", "c", "d", "e", "f"]
# choices = ["1", "2", "3", "4", "5", "6"]

llama_dir = "/gpfswork/rech/imi/utu57ed/llama/llama_files/"

high_level_cat_dict = {
    "Hedonism,Stimulation,Self-Direction": "Openness to Change",
    "Universalism,Benevolence": "Self-Transcendence",
    "Conformity,Tradition,Security": "Conservation",
    "Power,Achievement": "Self-Enhancement"
}


def get_prompt_skeleton(subject, experiment_name, args, simulated_participant):


    if not("pvq" in experiment_name or "hofstede" in experiment_name or "big5" in experiment_name or "mmlu" in experiment_name or "ultimatum" in experiment_name):
        raise NotImplementedError("Experiment name not implemented")

    # Natural language profile is only defined for dictionary profile
    if not args.profile_dict:
        if args.natural_language_profile:
            raise NotImplementedError("Natural language profile not implemented.")

    if args.format in ["code_py", "code_cpp", "conf_toml", "latex"]:
        if not args.no_profile:
            raise NotImplementedError(f"{args.format} format is only implemented with no_profile.")

    if args.weather:
        if not args.no_profile:
            raise NotImplementedError(f"{args.weather} is only implemented with no_profile.")

    if "pvq" in experiment_name:
        assert "pvq" in args.data_dir
        # todo: extract this somewhere to make it nicer
        test_name = "pvq"

    elif "ultimatum" in experiment_name:
        assert "ultimatum" in args.data_dir
        test_name = "ultimatum"

    elif "hofstede" in experiment_name:
        assert "hofstede" in args.data_dir
        test_name = "hofstede"

    elif "big5" in experiment_name:
        if args.data_dir == "data_big5_50":
            test_name = "big5_50"
        elif args.data_dir == "data_big5_100":
            test_name = "big5_100"
        else:
            raise ValueError(f"Data dir name is ill-defined {args.data_dir}")

    elif "mmlu" in experiment_name:
        assert "mmlu" in args.data_dir
        test_name = "mmlu"

    else:
        raise ValueError(f"Experiment name is ill-defined {args.experiment_name}")

    # setup the questionnaire description
    if test_name == "pvq":
        questionnaire_description = "Here we briefly describe some people. Please read each description and think about how much each person is or is not like you. Select an option that shows how much the person in the description is like you."
        questionnaire_description_empty = False

    elif test_name == "ultimatum":
        questionnaire_description = "In the following scenario, You have to decide whether to accept or reject a proposal."
        questionnaire_description_empty = False

    elif test_name == "hofstede":
        # VSM questionnaire doesn't have a description
        questionnaire_description = ""
        questionnaire_description_empty = True

    elif test_name == "mmlu":
        # VSM questionnaire doesn't have a description
        questionnaire_description = "The following are multiple choice questions (with answers)."
        questionnaire_description_empty = False

    elif test_name in ["big5_50", "big5_100"]:
        # VSM questionnaire doesn't have a description
        questionnaire_description = "Mark how much you agree with each statement."
        questionnaire_description_empty = False

    # setup the set_perspective_str
    if args.profile_dict:
        if args.query_prompt:
            raise NotImplementedError('as one letter not implemented for profile')

        if list(args.profile_dict.keys()) != ["Primary values"] and args.perspective_amount not in ["extreme", "slight", "more", "most"]:
            raise NotImplementedError('Perspective amount not implemented for keys other than "Primary values".')

        # source pvq: https://scholarworks.gvsu.edu/cgi/viewcontent.cgi?article=1116&context=orpc
        # source hofstede: https://geerthofstede.com/wp-content/uploads/2016/07/Manual-VSM-2013.pdf

        if args.natural_language_profile:

            # we define the profile in natural language
            assert set(args.profile_dict.keys()) == {'Primary values'}

            # extract primary values, add space after commas and replace the last comma with "and"
            primary_values_str = args.profile_dict["Primary values"]

            if args.add_high_level_categories:
                if test_name == "pvq":
                    primary_values_str += f",{high_level_cat_dict[args.profile_dict['Primary values']]}"
                else:
                    raise ValueError("High level categories are implemented only for pvq.")

            primary_values_str = primary_values_str.replace(",", ", ")
            primary_values_str = ", and ".join(primary_values_str.rsplit(', ', 1))

            if args.natural_language_profile_detail == "high":
                primary_values = args.profile_dict["Primary values"]
                if test_name == "pvq":
                    values_description = "Here are the explanations of those values:\n" + \
                                         ("\t- Self-Direction : independent thought and action - choosing, creating, exploring.\n" if "Self-Direction" in primary_values else "") + \
                                         ("\t- Stimulation : excitement, novelty, and challenge in life.\n" if "Stimulation" in primary_values else "") + \
                                         ("\t- Hedonism : pleasure or sensuous gratification for oneself.\n" if "Hedonism" in primary_values else "") + \
                                         ("\t- Achievement : personal success through demonstrating competence according to social standards.\n" if "Achievement" in primary_values else "") + \
                                         ("\t- Power : social status and prestige, control or dominance over people and resources.\n" if "Power" in primary_values else "") + \
                                         ("\t- Security : safety, harmony, and stability of society, of relationships, and of self.\n" if "Security" in primary_values else "") + \
                                         ("\t- Conformity : restraint of actions, inclinations, and impulses likely to upset or harm others and violate social expectations or norms.\n" if "Conformity" in primary_values else "") + \
                                         ("\t- Tradition : respect, commitment, and acceptance of the customs and ideas that one's culture or religion provides.\n" if "Tradition" in primary_values else "") + \
                                         ("\t- Benevolence : preserving and enhancing the welfare of those with whom one is in frequent personal contact (the ‘in-group’).\n" if "Benevolence" in primary_values else "") + \
                                         ("\t- Universalism : understanding, appreciation, tolerance, and protection for the welfare of all people and for nature.\n" if "Universalism" in primary_values else "")

                elif test_name == "hofstede":
                    values_description = "Here are the explanations of those values:\n" + \
                                         ("\t- Power distance: Power Distance is defined as the extent to which the less powerful members of institutions and organizations within a society expect and accept that power is distributed unequally.\n" if "" in primary_values else "Power distance") + \
                                         ("\t- Individualism vs Collectivism: Individualism is the opposite of Collectivism. Individualism stands for a society in which the ties between individuals are loose: a person is expected to look after himself or herself and his or her immediate family only. Collectivism stands for a society in which people from birth onwards are integrated into strong, cohesive in-groups, which continue to protect them throughout their lifetime in exchange for unquestioning loyalty.\n" if "Individualism" in primary_values else "") + \
                                         ("\t- Masculinity vs Femininity: Masculinity is the opposite of Femininity. Masculinity stands for a society in which social gender roles are clearly distinct: men are supposed to be assertive, tough, and focused on material success; women are supposed to be more modest, tender, and concerned with the quality of life. Femininity stands for a society in which social gender roles overlap: both men and women are supposed to be modest, tender, and concerned with the quality of life.\n"  if "Masculinity" in primary_values else "") + \
                                         ("\t- Uncertainty avoidance: Uncertainty Avoidance is defined as the extent to which the members of institutions and organizations within a society feel threatened by uncertain, unknown, ambiguous, or unstructured situations.\n" if "Uncertainty avoidance" in primary_values else "") + \
                                         ("\t- Short vs Long-term orientation: Long Term Orientation is the opposite of Short Term Orientation. Long Term Orientation stands for a society which fosters virtues oriented towards future rewards, in particular adaptation, perseverance and thrift. Short Term orientation stands for a society which fosters virtues related to the past and present, in particular respect for tradition, preservation of “face”, and fulfilling social obligations. \n" if "Long-term orientation" in primary_values else "")  + \
                                         ("\t- Indulgence vs Restraint: Indulgence stands for a society which allows relatively free gratification of some desires and feelings, especially those that have to do with leisure, merrymaking with friends, spending, consumption and sex. Its opposite pole, Restraint, stands for a society which controls such gratification, and where people feel less able to enjoy their lives.\n" if "Indulgence" in primary_values else "")

                elif test_name in ["big5_50", "big5_100", "ultimatum"]:
                    raise NotImplementedError(f"Descriptions not implemented for {test_name}.")

                else:
                    raise NotImplementedError(f"High detail option not implemented for {test_name}")

            elif args.natural_language_profile_detail == "no":
                values_description = ""
            else:
                raise ValueError(f"Variable natural_language_profile_detail is ill-defined ({args.natural_language_profile_detail}).")

            # perspective amount
            # todo: bug meidum (most) vs expterem (expremely more) -> not clear
            if args.perspective_amount == "extreme":
                amount_str = "extremely more importance"

            elif args.perspective_amount == "slight":
                amount_str = "slightly more importance"

            elif args.perspective_amount == "more":
                amount_str = "more importance"

            elif args.perspective_amount == "most":
                raise DeprecationWarning("Use 'more' instead.")
                amount_str = "the most importance"

            elif args.perspective_amount == "medium":
                raise DeprecationWarning("Use 'most' instead.")
            else:
                raise ValueError("Unrecognized perspective amount {args.perspective_amount}.")

            if args.direct_perspective:
                if args.pretend:
                    set_perspective_str = f"You pretending to be a person attributing {amount_str} to high {primary_values_str}."
                else:
                    set_perspective_str = f"You are a person attributing {amount_str} to high {primary_values_str}."

                if values_description:
                    set_perspective_str += f" {values_description}"
                else:
                    set_perspective_str += "\n"

            else:
                assert not args.pretend
                set_perspective_str = f"The following is a questionnaire (with answers) given to a person attributing {amount_str} to high {primary_values_str}."
                if values_description:
                    set_perspective_str += f" {values_description}"
                else:
                    set_perspective_str += "\n"

        else:

            if args.perspective_amount in ["extreme", "slight"]:
                amount_str = f"({args.perspective_amount})"
            elif args.perspective_amount in ["most", "more"]:
                raise NotImplementedError(f"{args.perspective_amount} not implemented with kv profile.")
            else:
                amount_str = ""

            # we define the perspective through a key:value profile

            # only using profile for primary values at the moment -> todo:remove this assert later
            assert set(args.profile_dict.keys()) == {'Primary values'}  # we add ":high "
            kv_profile_string = "\n".join([
                f"\t{k}{amount_str}:high {v}" for k, v in args.profile_dict.items()
            ])
            if args.add_high_level_categories:
                kv_profile_string += f",{high_level_cat_dict[args.profile_dict['Primary values']]}"

            if args.direct_perspective:
                set_perspective_str = "You are a person with the following profile:\n" + \
                                  f"{kv_profile_string}\n"

            else:
                set_perspective_str = "The following is a questionnaire (with answers) given to a person with the following profile:\n" + \
                                  f"{kv_profile_string}\n"

        # query string
        if args.direct_perspective:
            query_str = "Answer:"
        else:
            query_str = "Answer (from the person):"

    elif args.no_profile:

        if args.simulated_population_type in ["famous_people"]:
            set_perspective_str = f"You are {simulated_participant}"
        elif args.simulated_population_type in ["lotr_characters"]:
            set_perspective_str = f"You are {simulated_participant} from the Lord of The Rings."
        elif args.simulated_population_type in ["tolkien_characters"]:
            set_perspective_str = f"You are {simulated_participant} from J. R. R. Tolkien's Middle-earth legendarium."
        elif args.simulated_population_type == "anes":
            set_perspective_str = f"You are a person with the following profile.\n\n'''\n{simulated_participant}\n'''"
        elif args.simulated_population_type == "llm_personas":
            set_perspective_str = f"You are a person with the following profile.\n\n'''\n{simulated_participant}\n'''"
        elif args.simulated_population_type == "user_personas":
            set_perspective_str = f"You are talking to a person with the following profile.\n\n'''\n{simulated_participant}\n'''"
        elif args.simulated_population_type == "permutations":
            set_perspective_str = ""
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
            set_perspective_str = set_perspective_str.rstrip() + f"\n{weather_dict[args.weather]}"

        if args.format == "code_py":
            query_str = """# Choose the answer\nanswer = answers_dict[\""""

            questionnaire_description = f"""query = \"\"\"\n{questionnaire_description}"""

        elif args.format == "code_cpp":
            query_str = "\t// Choose the answer\n\tstd::string answer = answers_dict[\""

            questionnaire_description = "#include <iostream>\n" + \
            "#include <string>\n" + \
            "#include <map>\n" + \
            "int main() {\n" + \
            "\tstd::string query = R\"(\n" + \
            questionnaire_description

        elif args.format == "conf_toml":
            query_str = "answer = "

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
                query_str = "Answer:"

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
                query_str = "Answer:"

        else:
            raise ValueError(f"Undefined format {args.format}.")

    else:
        raise ValueError("Undefined perspective.")

    # only no_profile with code formats change the questionnaire description
    assert (not questionnaire_description_empty == questionnaire_description) or (args.no_profile and args.format != "chat")

    if args.system_message:
        prompts = {
            "system": f"{set_perspective_str}".rstrip(),  # remove newline from the end
            "intro": f"{questionnaire_description}\n\n" if not questionnaire_description_empty else questionnaire_description,
            "query": f"{query_str}",
        }

    else:
        if args.separator and not args.direct_perspective:
            set_perspective_str += "\n" + "-"*200

        prompts = {
            "intro": (f"{set_perspective_str}\n\n" if set_perspective_str else "") +
                     (f"{questionnaire_description}\n\n" if not questionnaire_description_empty else questionnaire_description),  # if questionnaire_description is empty don't add newlines
            "query": f"{query_str}",
        }

    return prompts


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
    prompt = df.iloc[idx, 0]  # add question to prompt
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
    
    if args.format in ["code_py"]:
        prompt += "\n\"\"\"\n\n"

        prompt += "# Define the answers dictionary\n"
        prompt += "answers_dict = {\n"

        for ch in choices[:num_options]:

            prompt += "\t\"{}.\": \"{}\",\n".format(ch, options_strings[permutations_dict[ch]])
        prompt += "}\n"

    elif args.format in ["code_cpp"]:
        prompt += "\n)\";\n\n"

        prompt += "\t// Define the answers dictionary\n"
        prompt += "\tstd::map<std::string, std::string> answers_dict = {\n"

        for ch in choices[:num_options]:
            prompt += "\t\t{\""+ch+".\", \""+options_strings[permutations_dict[ch]]+"\"},\n"
        prompt += "\t};\n"

    elif args.format in ["conf_toml"]:

        prompt = prompt.replace("\n", "\n# ")
        prompt = f"# {prompt}"

        for ch in choices[:num_options]:
            prompt += f"\n# {ch}. {options_strings[permutations_dict[ch]]}"

    elif args.format in ["latex"]:
        prompt += "\n\\begin{enumerate}[label=\\Alph*.]\n"

        for ch in choices[:num_options]:
            prompt += f"\t\\item {options_strings[permutations_dict[ch]]}\n"

        prompt += "\\end{enumerate}"

    else:
        for ch in choices[:num_options]:
            prompt += "\n{}. {}".format(ch, options_strings[permutations_dict[ch]])

    prompt_skeleton = get_prompt_skeleton(subject=subject, experiment_name=experiment_name, args=args, simulated_participant=simulated_participant)

    # query_in_reply will put query in the models response, if not add it to prompt here
    if not args.query_in_reply:
        prompt += "\n"+prompt_skeleton["query"]

    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])

    return prompt, num_options, prompt_skeleton


# def gen_prompt(train_df, subject, experiment_name, args, permutations_dict, simulated_participant, k=-1):
#
#     # get intro prompt (ex. "The following are .... \n\n" )
#     prompt = get_prompt_skeleton(subject=subject, experiment_name=experiment_name, args=args, simulated_participant=simulated_participant)["intro"]
#
#     if k == -1:
#         k = train_df.shape[0]
#     for i in range(k):
#         example_prompt, _, _ = format_example(train_df, i, subject=subject, experiment_name=experiment_name, args=args, permutations_dict=permutations_dict, simulated_participant=simulated_participant)
#         prompt += example_prompt
#     return prompt


def hash_chat_conv(msgs_conv):
    json_string = json.dumps(msgs_conv)

    # Create a SHA256 hash of the string
    hash_object = hashlib.sha256(json_string.encode())

    # Get the hexadecimal representation of the hash
    hex_dig = hash_object.hexdigest()

    return hex_dig



def eval(args, subject, engine, dev_df, test_df, permutations_dict, llm_generator=None, simulated_participant=None):
    cors = []
    all_probs = []
    all_answers = []
    all_generations = []

    # hashing for simulated conversations
    messages_conv = None
    messages_conv_hash = None

    gpt_token_counter = 0

    if args.simulate_conversation_theme and not args.system_message:
        raise NotImplementedError("simulated conversation is only implemented with system message.")

    for i in range(test_df.shape[0]):
        if i % 10 == 0:
            print(f"Eval progress: {i}/{test_df.shape[0]}")
        #  e.g. A -> A little like me
        label_2_text_option_dict = {
            label: test_df.iloc[i, score+1] for label, score in permutations_dict.items()
        }

        # get prompt and make sure it fits
        k = args.ntrain
        prompt_end, n_options, prompt_skeleton = format_example(
            test_df, i,
            subject=subject,
            experiment_name=args.experiment_name,
            include_answer=False,
            args=args,
            permutations_dict=permutations_dict,
            simulated_participant=simulated_participant
        )

        if k == 0:
            prompt_intro = prompt_skeleton['intro']

        else:
            raise DeprecationWarning("In-context examples are not supported.")
            # prompt_intro = gen_prompt(
            #     dev_df, subject,
            #     experiment_name=args.experiment_name,
            #     k=k,
            #     args=args,
            #     permutations_dict=permutations_dict,
            #     simulated_participant=simulated_participant
            # )

        prompt = prompt_intro + prompt_end

        # all questions have the same number of options
        assert test_df.shape[1]-2 == n_options
        answers = choices[:n_options]


        label = test_df.iloc[i, test_df.shape[1]-1]
        assert label in answers + ["undef"]

        if args.estimate_gpt_tokens:
            encoder = tiktoken.encoding_for_model('gpt-3.5-turbo-0301')
            assert encoder == tiktoken.encoding_for_model('gpt-4-0314')
            gpt_token_counter += len(encoder.encode(prompt_skeleton.get("system", "")+prompt)) + 1  # prompt + 1 generated token

        if engine == "dummy":
            if args.system_message:
                print("SYSTEM:", prompt_skeleton.get("system", None))

            print("PROMPT:", prompt)
            if args.query_in_reply:
                print(prompt_skeleton['query'])

            generation = random.choice([f"{c}. ba" for c in answers])
            lprobs = dummy_lprobs_from_generation(generation, answers, label_2_text_option_dict)

        elif engine == "interactive":
            # ask the user to choose
            if args.system_message:
                print("SYSTEM:", prompt_skeleton.get("system", None))

            if args.query_in_reply:
                prompt += f"{prompt_skeleton['query']} "

            generation = input(f"{prompt}")

            lprobs = dummy_lprobs_from_generation(generation, answers, label_2_text_option_dict)

        elif engine in ["gpt-3.5-turbo-0301", "gpt-4-0314", "gpt-3.5-turbo-0613", "gpt-4-0613", "gpt-3.5-turbo-1106-preview", "gpt-4-1106-preview"]:

            while True:
                try:
                    if args.query_in_reply:
                        raise ValueError("Can't use query_in_reply with gpt")

                    if args.system_message:
                        model_system_message = prompt_skeleton.get("system", None)

                        # system message
                        if args.simulate_conversation_theme:

                            if messages_conv is None:
                                print("Simulating conversation")
                                messages_conv, messages_conv_hash = simulate_conversation(
                                    args=args,
                                    engine=engine,
                                    sim_engine=engine,
                                    model_system_message=model_system_message,
                                )
                            else:
                                print("Loading cached conversation")
                                assert hash_chat_conv(messages_conv) == messages_conv_hash

                            messages = []
                            if model_system_message:
                                messages.append({"role": "system", "content": model_system_message})

                            messages = messages + messages_conv + [{"role": "user", "content": prompt}]

                            # estimate tokens
                            encoder = tiktoken.encoding_for_model(engine)
                            print("n_tokens:", len(encoder.encode(" ".join([m["content"] for m in messages[:-1]]))))

                        else:
                            messages = []
                            if model_system_message:
                                messages.append({"role": "system", "content": model_system_message})

                            messages.append({"role": "user", "content": prompt})

                    else:
                        if args.simulate_conversation_theme:
                            raise NotImplementedError("noisy conversation not implemented for user message")

                        # user message
                        messages = [
                            {"role": "user", "content": prompt}
                        ]

                    print_chat_messages(messages)
                    if args.generative_qa:
                        c = openai.ChatCompletion.create(
                            model=engine,
                            messages=messages,
                            max_tokens=args.max_tokens,
                            n=1,
                            # temperature=1,
                            request_timeout=30,
                            top_p=0.9,
                        )
                        generation = c['choices'][0]['message']['content']

                    else:
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

        elif engine in ["gpt-3.5-turbo-instruct-0914", "text-davinci-003", "text-davinci-002", "text-davinci-001", "curie", "babbage", "ada"]:

            if args.generative_qa:
                raise NotImplementedError("Generative QA not implemented for OpenAI non-ChatGPT models.")

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
            option_scores = {
                ans: output_dict.get(ans, -100) for ans in answers
            }

            # take the most probable answer as the generation
            generation = max(option_scores, key=option_scores.get)

            # extract logprobs
            lprobs = [float(option_scores[a]) for a in answers]

        elif engine in ["zephyr-7b-beta"] or "Mistral-7B" in engine or "Mixtral" in engine:

            tokenizer, model = llm_generator

            if args.system_message:
                if "Mistral-7B-instruct" in engine:
                    raise ValueError("Mistral does not support system messages")

                model_system_message = prompt_skeleton.get("system", None)

                if args.simulate_conversation_theme:

                    if messages_conv is None:
                        print("SIMULATING CONVERSATION")
                        messages_conv, messages_conv_hash = simulate_conversation(
                            args=args,
                            engine=engine,
                            sim_engine=engine,
                            model_system_message=model_system_message,
                            llm_generator=llm_generator
                        )
                    else:
                        print("LOADING CACHED CONVERSATION")
                        assert hash_chat_conv(messages_conv) == messages_conv_hash

                    messages = []
                    if model_system_message:
                        messages.append({"role": "system", "content": model_system_message})

                    messages = messages + messages_conv + [{"role": "user", "content": prompt}]

                else:
                    messages = []
                    if model_system_message:
                        messages.append({"role": "system", "content": model_system_message})

                    messages.append({"role": "user", "content": prompt})

            else:
                if args.simulate_conversation_theme:
                    raise NotImplementedError("noisy conversation not implemented for user message")

                messages = [{"role": "user", "content": prompt}]

            formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            if args.query_in_reply:
                formatted_prompt += f"{prompt_skeleton['query']} "

            print(f"************************\nFORMATTED PROMPT:{formatted_prompt}\n******************")
            inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

            if args.generative_qa:
                output_seq = model.generate(
                    **inputs,
                    max_new_tokens=args.max_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_k=50,
                    top_p=0.95,
                    return_dict_in_generate=True,
                    output_scores=True
                )
                generation = tokenizer.decode(output_seq.sequences[0][len(inputs.input_ids[0]):])
                lprobs = dummy_lprobs_from_generation(generation, answers, label_2_text_option_dict)

            else:
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

        elif engine in ["openassistant_rlhf2_llama30b"]:
            if args.generative_qa:
                raise NotImplementedError("Generative QA not implemented for OpenAI non-ChatGPT models.")

            if args.simulate_conversation_theme:
                raise NotImplementedError("noisy conversation not implemented for user message")

            if args.system_message:
                prompt = f'<prefix>{prompt_skeleton["system"]}</prefix><human>{prompt}<bot>'

            else:
                # prompt = f"<prefix></prefix><human>{prompt}<bot>"
                prompt = f'<human>{prompt}<bot>'

            tokenizer, model = llm_generator

            inputs = tokenizer(prompt, return_tensors='pt').to('cuda')
            del inputs["token_type_ids"]
            start_time = time.time()
            output = model.generate(
                **inputs,
                max_new_tokens=1,
                do_sample=False,
                temperature=0.001,
                top_p=1.0,
                return_dict_in_generate=True,
                output_scores=True
            )
            print("inference time:", time.time()-start_time)

            option_scores, generation, lprobs = parse_hf_outputs(output=output, tokenizer=tokenizer, answers=answers)

            # todo: remove if assert doesn't fail
            option_scores_ = { ans: output.scores[0][0, tokenizer.convert_tokens_to_ids(ans)] for ans in answers }
            generation_ = max(option_scores, key=option_scores.get)
            lprobs_ = [float(option_scores[a]) for a in answers]

            assert (option_scores_, generation_, lprobs_) == (option_scores, generation, lprobs)

        elif engine in ["stablevicuna"]:
            # todo: combine with stablelm
            if args.simulate_conversation_theme:
                raise NotImplementedError("noisy conversation not implemented for user message")

            if args.generative_qa:
                raise NotImplementedError("Generative QA not implemented for OpenAI non-ChatGPT models.")

            if args.system_message:
                raise NotImplementedError("System message not implemented.")

            else:
                prompt = f"""\
                ### Human: {prompt}
                ### Assistant:\
                """

            tokenizer, model = llm_generator

            inputs = tokenizer(prompt, return_tensors='pt').to('cuda')
            output = model.generate(
                **inputs,
                max_new_tokens=1,
                do_sample=False,
                temperature=0.001,
                top_p=1.0,
                return_dict_in_generate=True,
                output_scores=True
            )
            if args.simulate_conversation_theme:
                raise NotImplementedError("noisy conversation not implemented for user message")

            option_scores, generation, lprobs = parse_hf_outputs(output=output, tokenizer=tokenizer, answers=answers)
            option_scores_ = { ans: output.scores[0][0, tokenizer.convert_tokens_to_ids(ans)] for ans in answers }
            generation_ = max(option_scores, key=option_scores.get)
            lprobs_ = [float(option_scores[a]) for a in answers]
            assert (option_scores_, generation_, lprobs_) == (option_scores, generation, lprobs)

        elif engine in [
            *[f"llama_2_{s}_chat" for s in ["7b", "13b", "70b"]],
            *[f"llama_2_{s}" for s in ["7b", "13b", "70b"]]
        ]:

            tokenizer, model = llm_generator

            if args.system_message:
                model_system_message = prompt_skeleton.get("system", None)

                if args.simulate_conversation_theme:

                    if messages_conv is None:
                        print("SIMULATING CONVERSATION")
                        messages_conv, messages_conv_hash = simulate_conversation(
                            args=args,
                            engine=engine,
                            sim_engine=engine,
                            model_system_message=model_system_message,
                            llm_generator=llm_generator
                        )
                    else:
                        print("LOADING CACHED CONVERSATION")
                        assert hash_chat_conv(messages_conv) == messages_conv_hash

                    messages = []
                    if model_system_message:
                        messages.append({"role": "system", "content": model_system_message})

                    messages = messages + messages_conv + [{"role": "user", "content": prompt}]

                else:
                    messages = []
                    if model_system_message:
                        messages.append({"role": "system", "content": model_system_message})

                    messages.append({"role": "user", "content": prompt})

            else:
                if args.simulate_conversation_theme:
                    raise NotImplementedError("noisy conversation not implemented for user message")

                messages = [
                    {"role": "user", "content": prompt}
                ]

            if "chat" in args.engine:
                formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            else:

                if args.custom_chat_template:
                    # Op 1: custom format
                    formatted_prompt = apply_chat_template_for_base_model(messages, add_generation_prompt=True)

                else:
                    # OP 2: monkey patch
                    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            if args.query_in_reply:
                formatted_prompt += f"{prompt_skeleton['query']} "

            print(f"************************\nFORMATTED PROMPT:{formatted_prompt}\n******************")

            inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
            if args.generative_qa:

                # string match
                output_seq = model.generate(
                    **inputs,
                    max_new_tokens=args.max_tokens,
                    do_sample=True,
                    top_p=0.9,
                    top_k=50,
                    temperature=0.6,
                    repetition_penalty=1.2,
                    num_beams=1,
                    return_dict_in_generate=True,
                    output_scores=True
                )
                generation = tokenizer.decode(output_seq.sequences[0][len(inputs.input_ids[0]):])
                lprobs = dummy_lprobs_from_generation(generation, answers, label_2_text_option_dict)

            else:

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

        elif engine in ["up_llama_60b_instruct", "up_llama2_70b_instruct_v2"]:
            if args.simulate_conversation_theme:
                raise NotImplementedError("noisy conversation not implemented for user message")

            if args.system_message:
                prompt = f"### System:\n{prompt_skeleton['system']}\n\n### User:\n{prompt}\n\n### Assistant:\n"
            else:
                prompt = f"### User:\n{prompt}\n\n### Assistant:\n"

            tokenizer, model = llm_generator

            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            if "token_type_ids" in inputs:
                del inputs["token_type_ids"]

            streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

            output = model.generate(
                **inputs,
                streamer=streamer,
                use_cache=True,
                max_new_tokens=1,
                # temperature=0.0001,
                do_sample=False,
                top_p=1.0,
                return_dict_in_generate=True,
                output_scores=True
            )
            # output_text = tokenizer.decode(output[0], skip_special_tokens=True)

            option_scores, generation, lprobs = parse_hf_outputs(output=output, tokenizer=tokenizer, answers=answers)

        elif engine in ["rp_incite_7b_instruct", "rp_incite_7b_chat"]:
            if args.system_message:
                raise NotImplementedError("System message not implemented for RedPajama.")

            if args.simulate_conversation_theme:
                raise NotImplementedError("noisy conversation not implemented for user message")

            tokenizer, model = llm_generator

            inputs = tokenizer(prompt, return_tensors='pt').to(model.device)

            output = model.generate(
                **inputs, max_new_tokens=1, do_sample=False, temperature=0.0001, top_p=1.0, top_k=1,
                return_dict_in_generate=True, output_scores=True
            )

            option_scores, generation, lprobs = parse_hf_outputs(output=output, tokenizer=tokenizer, answers=answers)
            option_scores_ = {
                ans: output.scores[0][0, tokenizer.convert_tokens_to_ids(ans)] for ans in answers
            }
            generation_ = max(option_scores, key=option_scores.get)
            lprobs_ = [float(option_scores[a]) for a in answers]
            assert (option_scores_, generation_, lprobs_) == (option_scores, generation, lprobs)

        elif engine in ["stablelm"]:
            if args.simulate_conversation_theme:
                raise NotImplementedError("noisy conversation not implemented for user message")

            if args.generative_qa:
                raise NotImplementedError("Generative QA not implemented for OpenAI non-ChatGPT models.")

            tokenizer, model = llm_generator

            class StopOnTokens(StoppingCriteria):
                def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
                    stop_ids = [50278, 50279, 50277, 1, 0]
                    for stop_id in stop_ids:
                        if input_ids[0][-1] == stop_id:
                            return True
                    return False

            if args.system_message:
                system_prompt = prompt_skeleton["system"]
                user_prompt = prompt
                prompt = f"<|SYSTEM|>{system_prompt}<|USER|>{user_prompt}<|ASSISTANT|>"

            else:
                # prompt = f"<|SYSTEM|><|USER|>{prompt}<|ASSISTANT|>"
                prompt = f"<|USER|>{prompt}<|ASSISTANT|>"

            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

            output = model.generate(
                **inputs,
                max_new_tokens=1,
                temperature=0.001,
                do_sample=False,
                stopping_criteria=StoppingCriteriaList([StopOnTokens()]),
                return_dict_in_generate=True,
                output_scores=True
            )

            option_scores, generation, lprobs = parse_hf_outputs(output=output, tokenizer=tokenizer, answers=answers)
            option_scores_ = { ans: output.scores[0][0, tokenizer.convert_tokens_to_ids(ans)] for ans in answers }
            generation_ = max(option_scores, key=option_scores.get)
            lprobs_ = [float(option_scores[a]) for a in answers]
            assert (option_scores_, generation_, lprobs_) == (option_scores, generation, lprobs)

        else:
            raise ValueError(f"Not recognized model {engine}.")

        probs = softmax(np.array(lprobs))
        pred = {i: c for i, c in enumerate(answers)}[np.argmax(lprobs)]
        cor = pred == label

        print(colored(f"Pred:{pred} (Generation:{generation}; Score: {map_choice_to_number(pred, permutations_dict)})", "green"))
        # print("Correct: ", cor)
        print("------------------")

        cors.append(cor)

        all_probs.append(probs)
        all_answers.append(pred)
        all_generations.append(generation)

    acc = np.mean(cors)
    cors = np.array(cors)

    all_probs = np.array(all_probs)
    print("Average accuracy {:.3f} - {}".format(acc, subject))

    if args.estimate_gpt_tokens:
        print("total GPT tokens used: {} for subject {}".format(gpt_token_counter, subject))
        print(f"\tgpt-4 ~ {0.04 *gpt_token_counter/1000:.4f} dollars".format(args))
        print(f"\tgpt-3.5 ~ {0.002 *gpt_token_counter/1000:.4f} dollars".format(args))
        print(f"\tdavinci ~ {0.02 *gpt_token_counter/1000:.4f} dollars".format(args))
        print(f"\tcurie ~ {0.002 *gpt_token_counter/1000:.4f} dollars".format(args))
        print(f"\tbabagge ~ {0.0005 *gpt_token_counter/1000:.4f} dollars".format(args))
        print(f"\tada ~ {0.0004 *gpt_token_counter/1000:.4f} dollars".format(args))

    return cors, acc, all_probs, all_answers, all_generations, gpt_token_counter

def remove_prefix(s, pref):
    if s.startswith(pref):
        return s[len(pref):]
    return s


def main(args):
    engine = args.engine
    print("Engine:", engine)

    subjects = sorted([f.split("_test.csv")[0] for f in os.listdir(os.path.join(args.data_dir, "test")) if "_test.csv" in f])

    # Logging
    if args.log:
        if not os.path.exists(args.save_dir):
            os.mkdir(args.save_dir)
        if not os.path.exists(os.path.join(args.save_dir, "results_{}".format(engine))):
            os.mkdir(os.path.join(args.save_dir, "results_{}".format(engine)))

    # dump results dir
    dump_results_dir = os.path.join(args.save_dir, "_".join([
        args.experiment_name,
        engine,
        args.data_dir,
        f"permutations_{args.permutations}",
        f"ntrain_{args.ntrain}" if args.ntrain > 0 else "",
        f"no_profile_{args.no_profile}" if args.no_profile else "",
        f"format_{args.format}",
        f"weather_{args.weather}" if args.weather else "",
        f"simulate_conv_{args.simulate_conversation_theme}" if args.simulate_conversation_theme else "",
        f"profile_{args.profile}" if args.profile else "",
        timestamp
    ]))
    os.makedirs(dump_results_dir, exist_ok=True)
    print("Savedir: ", dump_results_dir)

    # Data preparation
    if len(subjects) == 0:
        raise ValueError("No subjects found.")

    if "mmlu" in args.experiment_name:
        if __name__ == '__main__':
            assert any([t in args.experiment_name for t in [
                "mmlu_high_school",
                "mmlu_college"
            ]])

    if "mmlu_high_school" in args.experiment_name:
        assert "data_mmlu" in args.data_dir

        subjects_to_evaluate = [
            'high_school_biology',
            'high_school_chemistry',
            'high_school_computer_science',
            'high_school_european_history',
            'high_school_geography',
            'high_school_government_and_politics',
            'high_school_macroeconomics',
            'high_school_mathematics',
            'high_school_microeconomics',
            'high_school_physics',
            'high_school_psychology',
            'high_school_statistics',
            'high_school_us_history',
            'high_school_world_history'
        ]

        assert set(subjects_to_evaluate).issubset(subjects)
        subjects = subjects_to_evaluate

    if "mmlu_college" in args.experiment_name:
        assert "data_mmlu" in args.data_dir

        subjects_to_evaluate = [
            "college_biology",
            "college_chemistry",
            "college_computer_science",
            "college_mathematics",
            "college_medicine",
            "college_physics",
        ]
        assert set(subjects_to_evaluate).issubset(subjects)
        subjects = subjects_to_evaluate

        # assert all college subjects are taken
        # subjects = [s for s in subjects if "college" in s]

    if "data_pvq" == args.data_dir:
        assert "pvq" in args.experiment_name

        subjects_to_evaluate = [
            "pvq_male",
            # "pvq_female",
        ]
        assert set(subjects_to_evaluate).issubset(subjects)
        subjects = subjects_to_evaluate

    print("args:", args)
    print("subj:", subjects)

    gpt_tokens_total = 0


    if engine in ["zephyr-7b-beta"]:
        print("Loading zephyr-7b-beta")
        tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta", cache_dir=hf_cache_dir, device_map="auto")
        model = AutoModelForCausalLM.from_pretrained("HuggingFaceH4/zephyr-7b-beta", torch_dtype=torch.bfloat16, device_map="auto", cache_dir=hf_cache_dir)

        llm_generator = (tokenizer, model)

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
        model = AutoModelForCausalLM.from_pretrained(f"mistralai/{engine}", device_map="auto", cache_dir=hf_cache_dir, torch_dtype=torch.float16)

        llm_generator = (tokenizer, model)

    elif engine in [
        "Mixtral-8x7B-v0.1-4b",
        "Mixtral-8x7B-Instruct-v0.1-4b",
    ]:
        print(f"Loading {engine}")
        model_name = engine.rstrip("-4b")
        tokenizer = AutoTokenizer.from_pretrained(f"mistralai/{model_name}", cache_dir=hf_cache_dir, device_map="auto")
        model = AutoModelForCausalLM.from_pretrained(f"mistralai/{model_name}", device_map="auto", cache_dir=hf_cache_dir, load_in_4bit=True)

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

    elif engine in ["up_llama_60b_instruct", "up_llama2_70b_instruct_v2"]:

        if engine == "up_llama2_70b_instruct_v2":
            tokenizer = AutoTokenizer.from_pretrained("upstage/Llama-2-70b-instruct-v2", cache_dir=hf_cache_dir)
            model = AutoModelForCausalLM.from_pretrained(
                "upstage/Llama-2-70b-instruct-v2",
                device_map="auto",
                torch_dtype=torch.float16,
                load_in_8bit=True, cache_dir=hf_cache_dir,
                rope_scaling={"type": "dynamic", "factor": 2}  # allows handling of longer inputs
            )

        elif engine == "up_llama_60b_instruct":

            tokenizer = AutoTokenizer.from_pretrained("upstage/llama-65b-instruct", cache_dir=hf_cache_dir)
            model = AutoModelForCausalLM.from_pretrained(
                "upstage/llama-65b-instruct",
                device_map="auto",
                torch_dtype=torch.float16,
                load_in_8bit=True, cache_dir=hf_cache_dir,
                rope_scaling={"type": "dynamic", "factor": 2}  # allows handling of longer inputs
            )

        llm_generator = (tokenizer, model)


    elif engine in ["falcon-40b", "falcon-40b-instruct"]:
        raise NotImplementedError("Falcon not implemented.")
        model_name = f"tiiuae/{engine}"

        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=hf_cache_dir)
        model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=hf_cache_dir, trust_remote_code=True)
        # todo: internet needed -> problem with JZ

        pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        )
        sequences = pipeline(
            "Girafatron is obsessed with giraffes, the most glorious animal on the face of this Earth. Giraftron believes all other animals are irrelevant when compared to the glorious majesty of the giraffe.\nDaniel: Hello, Girafatron!\nGirafatron:",
            max_length=200,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
        )
        for seq in sequences:
            print(f"Result: {seq['generated_text']}")

    elif engine in ["rp_incite_7b_instruct", "rp_incite_7b_chat"]:

        MIN_TRANSFORMERS_VERSION = '4.25.1'
        # check transformers version
        assert transformers.__version__ >= MIN_TRANSFORMERS_VERSION, f'Please upgrade transformers to version {MIN_TRANSFORMERS_VERSION} or higher.'

        if engine == "rp_incite_7b_instruct":

            # init
            tokenizer = AutoTokenizer.from_pretrained("togethercomputer/RedPajama-INCITE-7B-Instruct", cache_dir=hf_cache_dir)
            model = AutoModelForCausalLM.from_pretrained("togethercomputer/RedPajama-INCITE-7B-Instruct", torch_dtype=torch.float16, cache_dir=hf_cache_dir)

            # tokenizer = AutoTokenizer.from_pretrained("togethercomputer/RedPajama-INCITE-7B-Instruct")
            # model = AutoModelForCausalLM.from_pretrained("togethercomputer/RedPajama-INCITE-7B-Instruct", torch_dtype=torch.float16)
            model = model.to('cuda:0')


        elif engine == "rp_incite_7b_chat":

            # init
            tokenizer = AutoTokenizer.from_pretrained("togethercomputer/RedPajama-INCITE-7B-Chat", cache_dir=hf_cache_dir)
            model = AutoModelForCausalLM.from_pretrained("togethercomputer/RedPajama-INCITE-7B-Chat", torch_dtype=torch.float16, cache_dir=hf_cache_dir)
            model = model.to('cuda:0')

        else:
            raise ValueError("Unknown model.")

        llm_generator = (tokenizer, model)


    elif engine in ["stablelm", "stablevicuna", "openassistant_rlhf2_llama30b"]:

        if engine == "stablelm":
            print("Loading stable-lm-tuned-alpha-7b.")
            tokenizer = AutoTokenizer.from_pretrained("StabilityAI/stablelm-tuned-alpha-7b", cache_dir=hf_cache_dir)
            model = AutoModelForCausalLM.from_pretrained("StabilityAI/stablelm-tuned-alpha-7b", cache_dir=hf_cache_dir)

        elif engine == "stablevicuna":
            print("Loading stable-vicuna-13b.")
            tokenizer = AutoTokenizer.from_pretrained("/gpfswork/rech/imi/utu57ed/hf_stable_vicuna_13b")
            model = AutoModelForCausalLM.from_pretrained("/gpfswork/rech/imi/utu57ed/hf_stable_vicuna_13b")

        elif engine == "openassistant_rlhf2_llama30b":
            print("Loading openassistant-rlhf2-llama30b.")
            tokenizer = AutoTokenizer.from_pretrained("/gpfswork/rech/imi/utu57ed/oasst-rlhf-2-llama-30b-7k-steps-xor/oasst-rlhf-2-llama-30b-7k-steps")
            print("tokenizer loaded")
            start_time = time.time()
            model = AutoModelForCausalLM.from_pretrained("/gpfswork/rech/imi/utu57ed/oasst-rlhf-2-llama-30b-7k-steps-xor/oasst-rlhf-2-llama-30b-7k-steps")
            print(f"Model loaded (time={time.time() - start_time})")

        else:
            raise NotImplementedError(f"{engine} not supported")

        model.half().cuda()
        llm_generator = (tokenizer, model)

    else:
        llm_generator = None

    print(f"Loaded model: {args.engine}.")

    if "hofstede" in args.data_dir:
        n_options = 5
    elif "pvq" in args.data_dir:
        n_options = 6
    elif "big5" in args.data_dir:
        n_options = 5
    elif "ultimatum" in args.data_dir:
        n_options = 2
    else:
        raise ValueError(f"Undefined number of options for data in {args.data_dir}.")

    # prepare permutations
    if args.simulated_population_type == "permutations" and args.permutations > 1:

        import itertools
        all_permutations=list(itertools.permutations(range(n_options)))

        original_state = random.getstate()  # save the original state
        random.seed(1)
        permutations = random.sample(all_permutations, args.permutations)
        random.setstate(original_state)
        print("permutations_hash:", np.array(permutations)[:, :3].sum())

        permutations_dicts = [
            dict(zip(choices, perm)) for perm in permutations
        ]

        # permutations_dict = {choices[len(choices)-1-i]: i for i, c in enumerate(choices)}  # reverse

    else:
        # in order
        permutations_dicts = [{choices[i]: i for i, c in enumerate(choices[:n_options])}]

    # prepare simulated the simulated populations : personas/characters/permutations
    if args.simulated_population_type == "permutations":
        # dummy participant, population comes from permutation dicts
        simulated_population = [None]*len(permutations_dicts)

    else:
        if args.simulated_population_type == "lotr_characters": # todo: refactor: repetitive code below

            # https://www.quora.com/Can-you-name-the-50-most-mentioned-characters-in-the-Lord-of-the-Rings-trilogy
            # Smeagole and Gollum were separated and Gloin was removed so that we have 50 characters
            with open("personas/lotr_characters.txt") as f:
                simulated_population = [name.rstrip() for name in f.readlines()]

        elif args.simulated_population_type == "tolkien_characters":
            
            # https://en.wikipedia.org/wiki/List_of_Middle-earth_characters
            # 50 characters with the longest wikipedia page
            with open("personas/tolkien_characters.txt") as f:
                simulated_population = [name.rstrip() for name in f.readlines()]

        elif args.simulated_population_type == "anes":
            # https://electionstudies.org/
            with open("personas/ANES/voters2016_50.json") as f:
                simulated_population = json.load(f)

        elif args.simulated_population_type == "famous_people":
            # source: https://www.biographyonline.net/people/famous-100.html
            with open("personas/famous_people.txt") as f:
                simulated_population = [name.rstrip() for name in f.readlines()]

        elif args.simulated_population_type in ["llm_personas", "user_personas"]:
            with open("personas/personachat/grammar/personachat_I.txt") as f:
                simulated_population = [bio.rstrip() for bio in f.readlines()]

        # always use the same permutation
        assert len(permutations_dicts) == 1
        permutations_dicts = permutations_dicts * len(simulated_population)

    all_cors = []

    # list because of permutations
    subj_acc = [{} for _ in range(len(simulated_population))]
    subj_len = [{} for _ in range(len(simulated_population))]
    metrics = [{} for _ in range(len(simulated_population))]
    answers = [{} for _ in range(len(simulated_population))]
    generations = [{} for _ in range(len(simulated_population))]

    # evaluate model
    for subject in subjects:

        # load and prepare data
        if args.ntrain >= 1:
            # create in-context examples from dev
            dev_df = pd.read_csv(
                os.path.join(args.data_dir, "dev", subject + "_dev.csv"),
                header=None,
                keep_default_na=False,
            )[:args.ntrain]
            # if the question contains \n in the csv it will get parsed as \\n, we revert it back here to be newline
            dev_df[0][:] = dev_df[0][:].str.replace("\\n", "\n")

        else:
            dev_df = None

        test_df = pd.read_csv(
            os.path.join(args.data_dir, args.eval_set, subject + f"_{args.eval_set}.csv"),
            header=None,
            keep_default_na=False,
        )
        # if the question contains \n in the csv it will get parsed as \\n, we revert it back here to be newline
        test_df[0][:] = test_df[0][:].str.replace("\\n", "\n")

        # evaluate over population
        for sim_part_i, (simulated_participant, permutations_dict) in enumerate(zip(simulated_population, permutations_dicts)):
            # for sim_part_i, permutations_dict in enumerate(permutations_dicts):
            print(f"Simulated participant {sim_part_i}")

            cors, acc, probs, preds, gens, gpt_tokens = eval(
                args=args,
                subject=subject,
                engine=engine,
                dev_df=dev_df,
                test_df=test_df,
                permutations_dict=permutations_dict,
                llm_generator=llm_generator,
                simulated_participant=simulated_participant,
            )
            all_cors.append(cors)
            gpt_tokens_total += gpt_tokens

            subj_acc[sim_part_i][subject] = acc
            subj_len[sim_part_i][subject] = len(test_df)
            preds_values = np.vectorize(map_choice_to_number)(preds, permutations_dict)
            answers[sim_part_i][subject] = list(zip(preds, map(int, preds_values)))
            generations[sim_part_i][subject] = gens

            if "hofstede" in args.data_dir:
                assert "hofstede" in args.experiment_name

                # preds_values_ = np.vectorize(map_choice_to_number)(preds, permutations_dict)
                # assert all(preds_values_ == preds_values)

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

                # pvq is evaluated by averaging scored based on different values
                # preds_values_ = np.vectorize(map_choice_to_number)(preds, permutations_dict)
                # assert all(preds_values_ == preds_values)

                profile_values_idx_json = os.path.join(os.path.join(args.data_dir, "raw"), "values.json")
                with open(profile_values_idx_json) as f:
                    profile_values_idx = json.load(f)

                profile_values_idx = {k: np.array(v)-1 for k, v in profile_values_idx.items() if k != "_comment"}

                metrics[sim_part_i][subject] = {}
                # metrics[sim_part_i][subject]["raw"] = {}
                # metrics[sim_part_i][subject]["centralized"] = {}

                # average_participant_answer = preds_values.mean()

                for profile_value, idxs in profile_values_idx.items():
                    # metrics[subject][profile_value] = preds_values[idxs].mean() - mean_values
                    metrics[sim_part_i][subject][profile_value] = preds_values[idxs].mean() # legacy: todo: remove and save those below
                    # metrics[sim_part_i][subject]["raw"][profile_value] = preds_values[idxs].mean()
                    # metrics[sim_part_i][subject]["centralized"][profile_value] = preds_values[idxs].mean() - average_participant_answer

            elif "ultimatum" in args.data_dir:
                assert "ultimatum" in args.experiment_name

                # 1 - Accept; 2 - Reject;
                metrics[sim_part_i][subject] = {
                    "Acceptance Rate": np.mean(preds_values == 1),
                }

            else:
                metrics[sim_part_i][subject] = {
                    "accuracy": subj_acc[sim_part_i][subject]
                }

            res_test_df = test_df.copy()
            res_test_df["{}_correct".format(engine)] = cors
            for j in range(probs.shape[1]):
                choice = choices[j]
                res_test_df["{}_choice{}_probs".format(engine, choice)] = probs[:, j]

            if args.log:
                res_test_df.to_csv(os.path.join(args.save_dir, "results_{}".format(engine), "{}.csv".format(subject)), index=None)


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
            json.dump(
            {
                "args": vars(args),
                **mean_subj_acc,
                **{
                    "average": weighted_acc
                },
                "metrics": mean_metrics,
                "per_permutation_metrics": metrics,  # legacy todo: remove and update var_viz
                "per_simulated_participant_metrics": metrics,
                "simulated_population": simulated_population,
                "generations": generations,
                "answers": answers,
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
    parser.add_argument("--ntrain", "-k", type=int, default=0)
    parser.add_argument("--data_dir", "-d", type=str, default="data")
    parser.add_argument("--save_dir", "-s", type=str, default="results/results_test")
    parser.add_argument("--experiment_name", "-n", type=str, default="")
    parser.add_argument("--engine", "-e", type=str, default="dummy")
    parser.add_argument("--format", type=str, default="chat", choices=[
        "chat", "code_py", "code_cpp", "conf_toml", "latex"])
    parser.add_argument("--weather", type=str, default=None)
    parser.add_argument('--profile', type=str, help='Profile definition in format "k:v;k:v;k:v", ex. "age:35;interests:reading books"')
    parser.add_argument("--generative-qa", "-gqa", action="store_true", help="Use generative question answering instead of MCQ.")
    parser.add_argument("--max-tokens", "-mt", type=int, default=10, help="How many tokens to generate in genreative-qa")
    parser.add_argument("--query-in-reply", action="store_true", help="Force the query string as the beginning of the model's reply.")
    parser.add_argument("--custom-chat-template", action="store_true")
    parser.add_argument("--query-prompt", "-qp", type=str, help="Use Answer(as ONE letter): where applicable.")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--system-message", "-sm", action="store_true")
    parser.add_argument("--direct-perspective", action="store_true")
    parser.add_argument("--cold-run", "-cr", action="store_true")
    parser.add_argument("--estimate-gpt-tokens", "-t", action="store_true")
    parser.add_argument("--eval-set", type=str, default="test", choices=["test", "val"])
    parser.add_argument("--natural-language-profile", "-nlp", action="store_true", help="If true a profile will be defined in natural language as opposed to key value pairs.")
    parser.add_argument("--natural-language-profile-detail", type=str, default=None, choices=["no", "high"])
    parser.add_argument("--perspective-amount", type=str, default="medium", choices=["extreme", "medium", "slight", "more", "most"])
    parser.add_argument("--no-profile", action="store_true")  # todo: remove mcq-context?
    parser.add_argument("--mcq-context", action="store_true")  # todo: remove mcq-context?
    parser.add_argument("--wiki-context", action="store_true")
    parser.add_argument("--context-type", type=str, default=None)
    # parser.add_argument("--add-noisy-conversation", action="store_true")
    parser.add_argument("--simulate-conversation-theme", type=str, default=None)
    parser.add_argument("--log", "-l", type=bool, default=False)  # doesn't work well for multiproc (bigger llama models) # remove this parameter?
    # parser.add_argument("--permutations", "-p", type=int, default=1)
    parser.add_argument("--permutations", "-p", type=int, default=1)
    parser.add_argument("--simulated-population-type", "-pop", type=str, default="permutations", choices=["permutations", "lotr_characters", "tolkien_characters", "famous_people", "llm_personas", "user_personas", "anes"])
    parser.add_argument("--separator", action="store_true")
    parser.add_argument("--add-high-level-categories", action="store_true")
    parser.add_argument("--pretend", action="store_true")
    args = parser.parse_args()

    profile = {}
    if args.profile:
        for item in args.profile.split(';'):
            key, value = item.split(':')
            profile[key] = value

        args.profile_dict = profile

        print(f"Profile:\n{profile}")

    else:
        args.profile_dict = None

    if args.simulate_conversation_theme in ["None", "none"]:
        args.simulate_conversation_theme = None

    if args.estimate_gpt_tokens:
        if "gpt" not in args.engine and args.engine != "dummy":
            raise ValueError("Only gpt-4 gpt-3 and dummy support estimating GPT tokens")

    if args.cold_run:
        print("2nd person:", args.direct_perspective)
        print("System message:", args.system_message)
        # just used to show the profile to be used
        exit()

    if not args.separator and not (args.no_profile or args.mcq_context or args.wiki_context):
        raise ValueError("You are not using a separator?")

    if ("gpt-3.5" in args.engine and args.permutations > 50) or ("gpt-4" in args.engine and args.permutations > 5):
        raise ValueError(f"Are you sure you want to use {args.permutations} with {args.engine}??")

    # assert for plosone gpt or query_in_reply for other models
    # because query_in_reply can't be implemented for GPTs
    assert args.query_in_reply or "gpt" in args.engine

    if "gpt" in args.engine:
        if args.query_in_reply:
            raise ValueError("Can't use query in reply with gpt models")


    if args.profile:
        assert args.natural_language_profile
        assert args.natural_language_profile_detail == "no"

    assert args.ntrain == 0

    # check that only one profile type is active
    assert sum(map(bool, [args.no_profile, args.profile])) == 1

    if "pvq" in args.data_dir and args.profile:
        assert args.add_high_level_categories

    main(args)

