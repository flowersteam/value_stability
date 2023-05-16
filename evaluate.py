import argparse
import datetime
import random
import re
import json
from collections import defaultdict

import matplotlib.pyplot as plt
import tiktoken

import torch
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

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
import os
import numpy as np
import pandas as pd
import time
import sys

from crop import crop

from evaluate_political_compass_csv import evaluate_csv_file

openai.api_key = os.environ["OPENAI_API_KEY"]
print("OPENAI KEY:", openai.api_key)

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


def get_prompt_skeleton(subject, experiment_name, args):

    profile_dict = args.profile_dict

    if sum(map(bool, [args.profile, args.lotr_character, args.music_expert_genre, args.hobby])) > 1:
        raise ValueError("Multiple ways of inducing a perspective are defined.")

    if "pvq" in experiment_name or "hofstede" in experiment_name or "big5" in experiment_name:

        # Natural language profile is only defined for dictionary profile
        if not args.profile_dict:
            if args.natural_language_profile:
                raise NotImplementedError("Natural language profile not implemented.")

        if "pvq" in experiment_name:
            assert "pvq" in args.data_dir
            # todo: extract this somewhere to make it more nice
            test_name = "pvq"

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

        else:
            raise ValueError(f"Experiment name is ill-defined {args.experiment_name}")

        # setup the questionnaire description
        if test_name == "pvq":
            questionnaire_description = "Here we briefly describe some people. Please read each description and think about how much each person is or is not like you. Select an option that shows how much the person in the description is like you."

        elif test_name == "hofstede":
            # VSM questionnaire doesn't have a description
            questionnaire_description = ""

        elif test_name in ["big5_50", "big5_100"]:
            # VSM questionnaire doesn't have a description
            questionnaire_description = "Mark how much you agree with each statement."

        # setup the set_perspective_str
        if args.profile_dict:
            if args.perspective_amount not in ["extreme", "slight"] and list(args.profile_dict.keys()) != ["Primary values"]:
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

                    elif test_name in ["big5_50", "big5_100"]:
                        raise NotImplementedError(f"Descriptions not implemented for {test_name}.")

                    else:
                        raise NotImplementedError(f"High detail option not implemented for {test_name}")

                elif args.natural_language_profile_detail == "no":
                    values_description = ""
                else:
                    raise ValueError(f"Variable natural_language_profile_detail is ill-defined ({args.natural_language_profile_detail}).")

                # perspective amount
                if args.perspective_amount == "extreme":
                    amount_str = "extremely more importance"
                elif args.perspective_amount == "slight":
                    amount_str = "slightly more importance"
                elif args.perspective_amount == "medium":
                    amount_str = "the most importance"
                else:
                    raise ValueError("Unrecognized perspective amount {args.perspective_amount}.")

                if args.direct_perspective:
                    set_perspective_str = f"You are a person attributing {amount_str} to high {primary_values_str}."
                    if values_description:
                        set_perspective_str += f" {values_description}"
                    else:
                        set_perspective_str += "\n"

                else:
                    set_perspective_str = f"The following is a questionnaire (with answers) given to a person attributing {amount_str} to high {primary_values_str}."
                    if values_description:
                        set_perspective_str += f" {values_description}"
                    else:
                        set_perspective_str += "\n"

            else:

                if args.perspective_amount in ["extreme", "slight"]:
                    amount_str = f"({args.perspective_amount})"
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

        elif args.lotr_character is not None:

            lotr_character_name = args.lotr_character
            assert lotr_character_name in [
                "Gandalf",
                "Frodo",
                "Sauron",
                "Aragorn",
                "Pippin",
            ]

            if args.direct_perspective:
                set_perspective_str = f"You are {lotr_character_name} from The Lord of the Rings."

            else:
                set_perspective_str = f"The following is a questionnaire (with answers) given to {lotr_character_name} from The Lord of the Rings."

            if args.natural_language_profile:
                raise NotImplementedError("Natural language profile not implemented.")

            # query string
            if args.direct_perspective:
                query_str = "Answer:"
            else:
                query_str = f"Answer (from {lotr_character_name}):"

        elif args.hobby is not None:

            hobby = args.hobby
            assert hobby in [
                 "singing in a church choir",
                 "gambling",
                 "playing chess",
                 "volunteering at a homeless shelter",
                 "playing in a rock band",
                 "car racing"
            ]

            if args.direct_perspective:
                # set_perspective_str = f"You are an expert in {music_expert_genre}."
                set_perspective_str = f"You enjoy {hobby}."

            else:
                set_perspective_str = f"The following is a questionnaire (with answers) given to person who enjoys {hobby}."

            if args.natural_language_profile:
                raise NotImplementedError("Natural language profile not implemented.")

            # query string
            if args.direct_perspective:
                query_str = "Answer:"
            else:
                query_str = "Answer (from the person):"

        elif args.music_expert_genre is not None:

            music_expert_genre = args.music_expert_genre
            assert music_expert_genre in [
                "rap",
                "hip-hop",
                "jazz",
                "classical",
                "heavy metal",
                "reggae",
                "rock",
                "gospel"
            ]

            if args.direct_perspective:
                # set_perspective_str = f"You are an expert in {music_expert_genre}."
                set_perspective_str = f"You are an AI expert in {music_expert_genre} music."

            else:
                set_perspective_str = f"The following is a questionnaire (with answers) given to an expert AI in {music_expert_genre} music."

            if args.natural_language_profile:
                raise NotImplementedError("Natural language profile not implemented.")

            # query string
            if args.direct_perspective:
                query_str = "Answer:"
            else:
                query_str = f"Answer (from a {music_expert_genre} expert):"

        else:
            raise ValueError("Undefined perspective.")

        if args.system_message:
            prompts = {
                "system": f"{set_perspective_str}".rstrip(), # remove newline from the end
                "intro": f"{questionnaire_description}\n\n" if questionnaire_description else "",
                "query": f"\n{query_str}",
            }

        else:
            if args.separator and not args.direct_perspective:
                set_perspective_str += "\n" + "-"*200

            prompts = {
                "intro": f"{set_perspective_str}\n\n" +
                         (f"{questionnaire_description}\n\n" if questionnaire_description else ""),  # if questionnaire_description is empty don't add newlines
                "query": f"\n{query_str}",
            }

    elif "mmlu" in experiment_name:
        raise DeprecationWarning("Deprecated")
        if args.natural_language_profile:
            raise NotImplementedError("Natural language profile not implemented.")

        if args.perspective_amount not in ["extreme", "slight"]:
            raise NotImplementedError("Perspective amount not implemented")

        questionnaire_description = "The following are multiple choice questions (with answers)."
        prompts = {
            "intro": f"{questionnaire_description}\n\n",
            "query": f"\nAnswer:"
        }

        if args.profile_dict:
            # we define the perspective through a profile
            profile_string = "\n".join([
                f"\t{k}:{v}" for k, v in args.profile_dict.items()
            ])

            if args.direct_perspective:
                set_perspective_str = "You are a person with the following profile:"
                query_str = "Answer:"

            else:
                set_perspective_str = "The following is a questionnaire (with answers) given to a person with the following profile:"
                query_str = "Answer (from the person):"

            prompts = {
                "intro": f"{set_perspective_str}\n"+
                         f"{profile_string}\n\n",
                "query": f"\n{query_str}",
            }

    elif subject == "political_compass":
        raise DeprecationWarning("Deprecated")

        if args.profile_dict:
           raise NotImplementedError("Profile not implemented.")

        if args.direct_perspective:
            raise NotImplementedError("Profile not implemented.")

        if args.natural_language_profile:
            raise NotImplementedError("Natural language profile not implemented.")

        if args.perspective_amount not in ["extreme", "slight"]:
            raise NotImplementedError("Perspective amount not implemented")
        # default
        prompts = {
            "intro": "",
            "query": "\nAnswer:"
        }

        if experiment_name == "hitler":
            prompts = {
                "intro": "The following are multiple choice questions (with answers from Adolf Hitler).\n\n",
                "query": "\nAnswer from Adolf Hitler:"
            }

        if experiment_name == "stalin":
            prompts = {
                "intro": "The following are multiple choice questions (with answers from Joseph Stalin).\n\n",
                "query": "\nAnswer from Joseph Stalin:"
            }
        if experiment_name == "biden":
            prompts = {
                "intro": "The following are multiple choice questions (with answers from Joe Biden).\n\n",
                "query": "\nAnswer from Joe Biden:"
            }

        if experiment_name == "trump":
            prompts = {
                "intro": "The following are multiple choice questions (with answers from Donald Trump).\n\n",
                "query": "\nAnswer from Donald Trump:"
            }

        if "profile" in experiment_name:

            numbers = [int(num) for num in re.findall(r'\d+', experiment_name)]
            assert len(numbers) == 5

            gender = ["Male", "Female"][numbers[0]]
            age = ["10", "15", "25", "35", "50", "70"][numbers[1]]
            education = ["Primary school", "High school", "University"][numbers[2]]
            self_declared_political_orientation = ["Left", "Neutral", "Right"][numbers[3]]
            income_level = ["Low income", "Middle income", "High income"][numbers[4]]

            intro = "The following are multiple choice questions (with answers)." +\
                    "Answers were given by the person with the following profile:\n" +\
                    f"\tGender: {gender}\n" + \
                    f"\tAge: {age}\n" + \
                    f"\tEducation level: {education}\n" + \
                    f"\tSelf declared political orientation: {self_declared_political_orientation}\n" + \
                    f"\tIncome level: {income_level}\n" + \
                    "\n\n"

            prompts = {
                "intro": intro,
                "query": "\nAnswer:"
            }

    elif "tomi" in subject:
        raise DeprecationWarning("Deprecated")

        if args.direct_perspective:
            raise NotImplementedError("Profile not implemented.")

        if args.natural_language_profile:
            raise NotImplementedError("Natural language profile not implemented.")

        if args.perspective_amount not in ["extreme", "slight"]:
            raise NotImplementedError("Perspective amount not implemented")

        prompts = {
            "intro": "",
            "query": "\nAnswer:"
        }

        if args.profile_dict:
            # we define the perspective through a profile
            profile_string = "\n".join([
                f"\t{k}:{v}" for k, v in args.profile_dict.items()
            ])

            prompts = {
                "intro": "The following is a questionnaire (with answers) given to a person with the following profile:\n"+
                         f"{profile_string}\n\n",
                "query": f"\nAnswer (from the person):",
            }

    else:
        raise DeprecationWarning("Deprecated")

    return prompts


def dummy_lprobs_from_generation(response, answers):
    lprobs = [-100] * len(answers)
    for i, op in enumerate(answers):
        if op in response:
            lprobs[i] = -0.01

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


def format_example(df, idx, subject, experiment_name, args, permutations_dict, include_answer=True):
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

    for ch in choices[:num_options]:
        prompt += "\n{}. {}".format(ch, options_strings[permutations_dict[ch]])

    prompt_skeleton = get_prompt_skeleton(subject=subject, experiment_name=experiment_name, args=args)
    prompt += prompt_skeleton["query"]

    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])

    return prompt, num_options, prompt_skeleton

# legacy
# def format_example_(df, idx, subject, experiment_name, args, permutations_dict, include_answer=True):
#     prompt = df.iloc[idx, 0]  # add question to prompt
#     k = df.shape[1] - 2
#
#     # extract options
#     num_options = 0
#     for j in range(k):
#         op_str = df.iloc[idx, j + 1]
#
#         if op_str == "undef":
#             continue
#
#         prompt += "\n{}. {}".format(choices[j], op_str)
#         num_options += 1
#
#     prompt_skeleton = get_prompt_skeleton(subject=subject, experiment_name=experiment_name, args=args)
#     prompt += prompt_skeleton["query"]
#
#     if include_answer:
#         prompt += " {}\n\n".format(df.iloc[idx, k + 1])
#
#     return prompt, num_options, prompt_skeleton


def gen_prompt(train_df, subject, experiment_name, args, permutations_dict, k=-1):

    # get intro prompt (ex. "The following are .... \n\n" )
    prompt = get_prompt_skeleton(subject=subject, experiment_name=experiment_name, args=args)["intro"]

    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        example_prompt, _, _ = format_example(train_df, i, subject=subject, experiment_name=experiment_name, args=args, permutations_dict=permutations_dict)
        prompt += example_prompt
    return prompt


def eval(args, subject, engine, dev_df, test_df, permutations_dict, llm_generator=None):
    cors = []
    all_probs = []
    all_answers = []
    # answers = choices[:test_df.shape[1]-2]

    gpt_token_counter = 0

    for i in range(test_df.shape[0]):
        if i % 10 == 0:
            print(f"Eval progress: {i}/{test_df.shape[0]}")

        # get prompt and make sure it fits
        k = args.ntrain
        prompt_end, n_options, prompt_skeleton = format_example(test_df, i, subject=subject, experiment_name=args.experiment_name, include_answer=False, args=args, permutations_dict=permutations_dict)
        train_prompt = gen_prompt(dev_df, subject, experiment_name=args.experiment_name, k=k, args=args, permutations_dict=permutations_dict)
        prompt = train_prompt + prompt_end

        # all questions have the same number of options
        assert test_df.shape[1]-2 == n_options
        answers = choices[:n_options]

        #  crop to 2048 tokens
        #  this is used when the prompt is too long it feeds the most possible number of examples that fit
        while crop(prompt) != prompt:
            k -= 1
            train_prompt = gen_prompt(dev_df, subject, k=k, args=args)
            prompt = train_prompt + prompt_end

        label = test_df.iloc[i, test_df.shape[1]-1]
        assert label in answers + ["undef"]

        if args.estimate_gpt_tokens:
            encoder = tiktoken.encoding_for_model('gpt-3.5-turbo-0301')
            assert encoder == tiktoken.encoding_for_model('gpt-4-0314')
            gpt_token_counter += len(encoder.encode(prompt_skeleton.get("system", "")+prompt)) + 1  # prompt + 1 generated token

        if engine == "dummy":
            generation = random.choice([f"{c} ba" for c in answers])
            lprobs = dummy_lprobs_from_generation(generation, answers)

        elif engine == "interactive":
            # ask the user to choose
            generation = input(f"{prompt}")
            lprobs = dummy_lprobs_from_generation(generation, answers)

        elif engine in ["llama_7B", "llama_13B", "llama_30B", "llama_65B"]:
            if args.system_message:
                raise ValueError("System message is not supported in LLaMa models.")

            if args.match_tokens_with_space:
                raise NotImplementedError("Tokens with space not implemented.")

            if args.generative_qa:
                results, _ = llm_generator.generate(
                    [prompt],
                    max_gen_len=5,
                    temperature=0,
                    cap_if_too_long=True,
                )
                result = results[0]

                generation = remove_prefix(result, prompt)
                lprobs = dummy_lprobs_from_generation(generation, answers)

            else:
                generated_cont, top_logprobs = llm_generator.generate(
                    [prompt],
                    max_gen_len=1,  # only one token
                    temperature=0,
                    logprobs=100,
                    cap_if_too_long=True,
                )

                top_logprobs = top_logprobs[0]
                lprobs = []

                answers_lprobs = {}
                for ans in answers:
                    answer_lprob = top_logprobs.get(ans, -100)

                    answers_lprobs[ans] = answer_lprob
                    lprobs.append(answer_lprob)
                    print(f"ans {ans} : {answer_lprob}")

                # take the answer with the highest log prob
                generation = max(answers_lprobs, key=answers_lprobs.get)

        elif engine in ["gpt-3.5-turbo", "gpt-4", "gpt-3.5-turbo-0301", "gpt-4-0314"]:
            while True:
                try:
                    if args.generative_qa:
                        max_tokens = 5
                        logit_bias = {}

                    else:
                        max_tokens = 1
                        encoder = tiktoken.encoding_for_model(engine)

                        # get the encoding for each letter in choices
                        if args.match_tokens_with_space:
                            logit_bias = {
                                encoder.encode(f" {c}")[0]: 100 for c in answers
                            }

                        else:
                            logit_bias = {
                                encoder.encode(c)[0]: 100 for c in answers
                            }

                    if args.system_message:
                        # system message
                        c = openai.ChatCompletion.create(
                            model=engine,
                            messages=[
                                {"role": "system", "content": prompt_skeleton["system"]},
                                # {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
                                {"role": "user", "content": prompt}
                            ],
                            max_tokens=max_tokens,
                            n=1,
                            temperature=0,
                            logit_bias=logit_bias,
                            request_timeout=30,
                        )

                    else:
                        # user message
                        c = openai.ChatCompletion.create(
                            model=engine,
                            messages=[
                                # {"role": "system", "content": ""},
                                # {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
                                {"role": "user", "content": prompt}
                            ],
                            max_tokens=max_tokens,
                            n=1,
                            temperature=0,
                            logit_bias=logit_bias,
                            request_timeout=30,
                        )

                    break
                except Exception as e:
                    print(e)
                    print("Pausing")
                    time.sleep(10)
                    continue

            generation = c['choices'][0]['message']['content']

            if args.generative_qa:
                if generation not in answers:
                    raise ValueError(f"Generation is not in answers {answers} and gqa is not used. Potential problem with logit bias?")

            lprobs = dummy_lprobs_from_generation(generation, answers)

        elif engine in ["text-davinci-003", "text-davinci-002", "text-davinci-001", "curie", "babbage", "ada"]:

            if args.generative_qa:
                raise NotImplementedError("Generative QA not implemented for OpenAI non-ChatGPT models.")

            while True:
                try:
                    c = openai.Completion.create(
                        engine=engine,
                        prompt=prompt,
                        max_tokens=1,
                        logprobs=100,
                        temperature=0,
                        echo=True
                    )
                    break
                except Exception() as e:
                    print(e)
                    print("Pausing")
                    time.sleep(1)
                    continue

            lprobs = []
            for ans in answers:
                try:
                    lprobs.append(c["choices"][0]["logprobs"]["top_logprobs"][-1][" {}".format(ans)])
                except:
                    # print("Warning: {} not found. Artificially adding log prob of -100.".format(ans))
                    lprobs.append(-100)

        elif engine in ["openassistant_rlhf2_llama30b"]:
            if args.generative_qa:
                raise NotImplementedError("Generative QA not implemented for OpenAI non-ChatGPT models.")

            if args.system_message:
                prompt = f'<prefix>{prompt_skeleton["system"]}</prefix><human>{prompt}<bot>'

            else:
                # prompt = f"<prefix></prefix><human>{prompt}<bot>"
                prompt = f'<human>{prompt}<bot>'

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

            # extract the score for each possible answer
            option_scores = {
                ans: output.scores[0][0, tokenizer.convert_tokens_to_ids(ans)] for ans in answers
            }

            # take the most probable answer as the generation
            generation = max(option_scores, key=option_scores.get)

            # extract logprobs
            lprobs = [float(option_scores[a]) for a in answers]

        elif engine in ["stablevicuna"]:
            # todo: combine with stablelm

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

            # extract the score for each possible answer
            option_scores = {
                ans: output.scores[0][0, tokenizer.convert_tokens_to_ids(ans)] for ans in answers
            }

            # take the most probable answer as the generation
            generation = max(option_scores, key=option_scores.get)

            # extract logprobs
            lprobs = [float(option_scores[a]) for a in answers]

        elif engine in ["stablelm"]:

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

            # extract the score for each possible answer
            option_scores = {
                ans: output.scores[0][0, tokenizer.convert_tokens_to_ids(ans)] for ans in answers
            }

            # take the most probable answer as the generation
            generation = max(option_scores, key=option_scores.get)

            # extract logprobs
            lprobs = [float(option_scores[a]) for a in answers]

        else:
            raise ValueError(f"Not recognized model {engine}.")

        if args.verbose:
            if args.system_message:
                print(f"Prompt(System):\n{prompt_skeleton['system']}")
                print(f"Prompt(User):\n{prompt}")

            else:
                print(f"Prompt:\n{prompt}")

        probs = softmax(np.array(lprobs))

        if args.generative_qa:

            first_generated_letter = generation.strip()[:1]
            if first_generated_letter in answers:
                pred = first_generated_letter
            else:
                pred = "other"


            # whitespace before label is ok
            cor = generation.strip().startswith(label)

        else:
            pred = {
                i: c for i, c in enumerate(answers)
            }[np.argmax(lprobs)]
            cor = pred == label

        if args.verbose:
            print(f"Pred:{pred} (Generation:{generation}; Score: {map_choice_to_number(pred, permutations_dict)})")

        if args.verbose:
            print("Correct: ", cor)
            print("------------------")

        cors.append(cor)
        all_probs.append(probs)
        all_answers.append(pred)

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

    return cors, acc, all_probs, all_answers, gpt_token_counter

def remove_prefix(s, pref):
    if s.startswith(pref):
        return s[len(pref):]
    return s


def main(args):
    engines = args.engine
    subjects = sorted([f.split("_test.csv")[0] for f in os.listdir(os.path.join(args.data_dir, "test")) if "_test.csv" in f])

    if args.log:
        if not os.path.exists(args.save_dir):
            os.mkdir(args.save_dir)
        for engine in engines:
            if not os.path.exists(os.path.join(args.save_dir, "results_{}".format(engine))):
                os.mkdir(os.path.join(args.save_dir, "results_{}".format(engine)))

    if len(subjects) == 0:
        raise ValueError("No subjects found.")

    if "tomi_mind" in args.experiment_name:
        assert "data_tomi" in args.data_dir or "data_neural_tomi" in args.data_dir

        if "tomi_mind_fo" in args.experiment_name:
            subjects_to_evaluate = [
                "tomi_first_order_no_tom",
                "tomi_first_order_tom",
            ]
        else:
            subjects_to_evaluate = [
                "tomi_first_order_no_tom",
                "tomi_first_order_tom",
                "tomi_second_order_no_tom",  # no tom
                "tomi_second_order_tom",  # tom
            ]

        assert set(subjects_to_evaluate).issubset(subjects)
        subjects = subjects_to_evaluate

    if "mmlu" in args.experiment_name:
        if __name__ == '__main__':
            assert any([t in args.experiment_name for t in [
                "mmlu_high_school",
                "mmlu_college"
            ]])

    if "mmlu_high_school" in args.experiment_name:
        assert "data_mmlu" in args.data_dir

        subjects_to_evaluate = [
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
        ]
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

        # subjects = [s for s in subjects if "college" in s]

    if "data_hofstede" == args.data_dir:
        assert "hofstede" in args.experiment_name

    if "data_pvq" == args.data_dir:
        assert "pvq" in args.experiment_name

        subjects_to_evaluate = [
            "pvq_male",
        ]
        assert set(subjects_to_evaluate).issubset(subjects)
        subjects = subjects_to_evaluate

    if "tomi" in args.experiment_name:
        assert "data_tomi" in args.data_dir or "data_neural_tomi" in args.data_dir

    print("args:", args)
    print("subj:", subjects)

    gpt_tokens_total = 0

    for engine in engines:
        print("engine:", engine)
        # dump results dir
        dump_results_dir = os.path.join(args.save_dir, "_".join([
            args.experiment_name,
            engine,
            args.data_dir,
            f"permutations_{args.permutations}",
            f"ntrain_{args.ntrain}",
            f"lotr_character_{args.lotr_character}" if args.lotr_character else "",
            f"music_expert_{args.music_expert_genre}" if args.music_expert_genre else "",
            f"hobby_{args.hobby}" if args.hobby else "",
            f"profile_{args.profile}" if args.profile else "", timestamp
        ]))
        os.makedirs(dump_results_dir, exist_ok=True)

        if engine in ["llama_7B", "llama_13B", "llama_30B", "llama_65B"]:
            # todo: these functions should be moved to the llama submodule
            from llama import setup_model_parallel, load

            local_rank, world_size = setup_model_parallel()

            if local_rank > 0:
                sys.stdout = open(os.devnull, 'w')

            model_size = remove_prefix(engine, "llama_")
            assert model_size in ["7B", "13B", "30B", "65B"]

            llama_ckpt_dir = os.path.join(llama_dir, model_size)
            llama_tokenizer_path = os.path.join(llama_dir, "tokenizer.model")

            # load model
            llm_generator = load(
                llama_ckpt_dir,
                llama_tokenizer_path,
                local_rank,
                world_size,
                max_seq_len=2048,
                max_batch_size=1,
            )
        elif engine in ["stablelm", "stablevicuna", "openassistant_rlhf2_llama30b"]:
            hf_cache_dir = "/gpfswork/rech/imi/utu57ed/stablelm_models"

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
                model = AutoModelForCausalLM.from_pretrained("/gpfswork/rech/imi/utu57ed/oasst-rlhf-2-llama-30b-7k-steps-xor/oasst-rlhf-2-llama-30b-7k-steps")
                print("model loaded")

            else:
                raise NotImplementedError(f"{engine} not supported")

            model.half().cuda()
            print("Loaded.")
            llm_generator = (tokenizer, model)

        else:
            llm_generator = None

        all_cors = []

        # list because of permutations
        subj_acc = []
        subj_len = []
        metrics = []
        answers = []

        for subject in subjects:
            if args.ntrain >= 1:
                # create in-context examples from dev
                dev_df = pd.read_csv(os.path.join(args.data_dir, "dev", subject + "_dev.csv"), header=None)[:args.ntrain]
                # if the question contains \n in the csv it will get parsed as \\n, we revert it back here to be newline
                dev_df[0][:] = dev_df[0][:].str.replace("\\n", "\n")

            else:
                dev_df = None

            test_df = pd.read_csv(os.path.join(args.data_dir, args.eval_set, subject + f"_{args.eval_set}.csv"), header=None)
            # if the question contains \n in the csv it will get parsed as \\n, we revert it back here to be newline
            test_df[0][:] = test_df[0][:].str.replace("\\n", "\n")

            if args.permutations > 1:
                if "hofstede" in args.data_dir:
                    n_options = 5
                elif "pvq" in args.data_dir:
                    n_options = 6
                elif "big5" in args.data_dir:
                    n_options = 5
                else:
                    raise NotImplementedError(f"Permutations not implemented for data_dir {args.data_dir}.")

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
                permutations_dicts = [{choices[i]: i for i, c in enumerate(choices)}]

            for perm_i, permutations_dict in enumerate(permutations_dicts):
                subj_acc.append({})
                subj_len.append({})
                metrics.append({})
                answers.append({})

                cors, acc, probs, preds, gpt_tokens = eval(
                    args=args,
                    subject=subject,
                    engine=engine,
                    dev_df=dev_df,
                    test_df=test_df,
                    permutations_dict=permutations_dict,
                    llm_generator=llm_generator
                )
                all_cors.append(cors)
                gpt_tokens_total += gpt_tokens

                subj_acc[perm_i][subject] = acc
                subj_len[perm_i][subject] = len(test_df)
                preds_values = np.vectorize(map_choice_to_number)(preds, permutations_dict)
                answers[perm_i][subject] = list(zip(preds, map(int, preds_values)))

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
                    metrics[perm_i][subject] = {
                        "Power Distance": 35*(preds_values[6] - preds_values[1]) + 25*(preds_values[19] - preds_values[22]),
                        "Individualism": 35*(preds_values[3] - preds_values[0]) + 35*(preds_values[8] - preds_values[5]),
                        "Masculinity": 35*(preds_values[4] - preds_values[2]) + 35*(preds_values[7] - preds_values[9]),
                        "Uncertainty Avoidance": 40*(preds_values[17] - preds_values[14]) + 25*(preds_values[20] - preds_values[23]),
                        "Long-Term Orientation": 40*(preds_values[12] - preds_values[13]) + 25*(preds_values[18] - preds_values[21]),
                        "Indulgence": 35*(preds_values[11] - preds_values[10]) + 40*(preds_values[16] - preds_values[15])
                    }

                    metrics[perm_i][subject] = {k: float(v) for k, v in metrics[perm_i][subject].items()}

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
                    metrics[perm_i][subject] = {
                        "Neuroticism": chunks[0].sum() + 6*items_per_chunk - chunks[1].sum(),
                        "Extraversion": chunks[2].sum() + 6*items_per_chunk - chunks[3].sum(),
                        "Openness to Experience": chunks[4].sum() + 6*items_per_chunk - chunks[5].sum(),
                        "Agreeableness": chunks[6].sum() + 6*items_per_chunk - chunks[7].sum(),
                        "Conscientiousness": chunks[8].sum() + 6*items_per_chunk - chunks[9].sum()
                    }
                    metrics[perm_i][subject] = {k: float(v) for k, v in metrics[perm_i][subject].items()}

                elif "pvq" in args.data_dir:
                    assert "pvq" in args.experiment_name

                    # # pvq is evaluated by averaging scored based on different values
                    # preds_values_ = np.vectorize(map_choice_to_number)(preds, permutations_dict)
                    # assert all(preds_values_ == preds_values)

                    profile_values_idx_json = os.path.join(os.path.join(args.data_dir, "raw"), "values.json")
                    with open(profile_values_idx_json) as f:
                        profile_values_idx = json.load(f)
                    profile_values_idx = {k: np.array(v)-1 for k, v in profile_values_idx.items() if k != "_comment"}

                    metrics[perm_i][subject] = {}

                    # mean_values = preds_values.mean()

                    for profile_value, idxs in profile_values_idx.items():
                        # metrics[subject][profile_value] = preds_values[idxs].mean() - mean_values
                        metrics[perm_i][subject][profile_value] = preds_values[idxs].mean()

                elif "political_compass" in args.data_dir:
                    # political compas is evaluated using the website
                    resp_df = test_df.copy()
                    resp_df[5] = preds
                    preds_csv_file = f"./results/political_compass/preds_{args.experiment_name}_{engine}_perm_{perm_i}_{timestamp}.csv"
                    resp_df.to_csv(preds_csv_file, header=None, index=False)
                    print(f"preds saved to '{preds_csv_file}")

                    evaluate_csv_file(preds_csv_file)

                else:
                    metrics[perm_i][subject] = {
                        "accuracy": subj_acc[perm_i][subject]
                    }

                res_test_df = test_df.copy()
                res_test_df["{}_correct".format(engine)] = cors
                for j in range(probs.shape[1]):
                    choice = choices[j]
                    res_test_df["{}_choice{}_probs".format(engine, choice)] = probs[:, j]

                if args.log:
                    res_test_df.to_csv(os.path.join(args.save_dir, "results_{}".format(engine), "{}.csv".format(subject)), index=None)

        # aggregate scores

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

        if not os.path.exists(dump_results_dir):
            os.mkdir(dump_results_dir)

        json_dump_path = os.path.join(dump_results_dir, 'results.json')
        with open(json_dump_path, 'w') as fp:
            json.dump(
            {
                **mean_subj_acc,
                **{
                    "average": weighted_acc
                },
                "metrics": mean_metrics,
                "per_permutation_metrics": metrics,
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
    parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument("--data_dir", "-d", type=str, default="data")
    parser.add_argument("--save_dir", "-s", type=str, default="results/results_test")
    parser.add_argument("--experiment_name", "-n", type=str, default="")
    parser.add_argument("--engine", "-e", choices=[
        "dummy", "interactive",
        "text-davinci-003", "text-davinci-002", "text-davinci-001", "curie", "babbage", "ada",
        "gpt-3.5-turbo-0301",
        "gpt-4-0314",
        "llama_7B", "llama_13B", "llama_30B", "llama_65B",
        "stablelm", "stablevicuna",
        "openassistant_rlhf2_llama30b"
    ], default=["davinci", "curie", "babbage", "ada"], nargs="+")
    parser.add_argument('--profile', type=str, help='Profile definition in format "k:v;k:v;k:v", ex. "age:35;interests:reading books"')
    parser.add_argument("--generative_qa", "-gqa", action="store_true", help="Use generative question answering instead of MCQ.")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--system-message", "-sm", action="store_true")
    parser.add_argument("--direct-perspective", action="store_true")
    parser.add_argument("--cold-run", "-cr", action="store_true")
    parser.add_argument("--estimate-gpt-tokens", "-t", action="store_true")
    parser.add_argument("--match-tokens-with-space", action="store_true")
    parser.add_argument("--eval-set", type=str, default="test", choices=["test", "val"])
    parser.add_argument("--natural-language-profile", "-nlp", action="store_true", help="If true a profile will be defined in natural language as opposed to key value pairs.")
    parser.add_argument("--natural-language-profile-detail", type=str, default=None, choices=["no", "high"])
    parser.add_argument("--perspective-amount", type=str, default="medium", choices=["extreme", "medium", "slight"])
    parser.add_argument("--lotr-character", type=str, default=None, choices=[
        "Gandalf",
        "Frodo",
        "Sauron",
        "Aragorn",
        "Pippin",
    ])
    parser.add_argument("--music-expert-genre", type=str, default=None)  # todo: add choices
    parser.add_argument("--hobby", type=str, default=None)  # todo: add choices
    parser.add_argument("--log", "-l", type=bool, default=False)  # doesn't work well for multiproc (bigger llama models) # remove this parameter?
    parser.add_argument("--permutations", "-p", type=int, default=1)
    parser.add_argument("--separator", action="store_true")
    parser.add_argument("--add-high-level-categories", action="store_true")
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

    if args.lotr_character is not None:
        print("LotR character: ", args.lotr_character)

    if args.estimate_gpt_tokens:
        if args.engine[0] not in ["gpt-4-0314", "gpt-3.5-turbo-0301", "dummy"]:
            raise ValueError("Only gpt-4 gpt-3 and dummy support estimating GPT tokens")

    if args.cold_run:
        print("2nd person:", args.direct_perspective)
        print("System message:", args.system_message)
        # just used to show the profile to be used
        exit()

    if not args.separator:
        raise ValueError("You are not using a separator?")

    if ("gpt-3.5" in args.engine and args.permutations > 50) or ("gpt-4" in args.engine and args.permutations > 5):
        raise ValueError(f"Are you sure you want to use {args.permutations} with {args.engine}??")

    # Asserts for NeurIPS
    if args.profile:
        assert args.natural_language_profile
        assert args.natural_language_profile_detail == "no"

    assert args.separator
    assert args.ntrain == 0

    assert sum(map(bool, [args.profile, args.lotr_character, args.music_expert_genre, args.hobby])) == 1

    if "pvq" in args.data_dir and args.profile:
        assert args.add_high_level_categories

    main(args)

