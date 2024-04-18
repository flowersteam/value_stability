import argparse
import random
from collections import defaultdict
import os
import json
import hashlib
import time
import datetime
import itertools
import string

from termcolor import colored

from utils import *


import numpy as np
import pandas as pd
import torch
import tiktoken

from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList, BitsAndBytesConfig
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training

from personas.utils import simulated_participant_to_name

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff


@retry(wait=wait_random_exponential(min=10, max=30), stop=stop_after_attempt(10))
def completions_with_backoff(client, **kwargs):
    return client.chat.completions.create(**kwargs)


hf_cache_dir = get_hf_cache_dir()
os.environ['TRANSFORMERS_CACHE'] = hf_cache_dir


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
        if args.verbose:
            print(f"Simulted conv msg {msg_i}")

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
            response = f"Dummy simulated message no. {msg_i}. This is a filler message it same some extra text so as to help estimate the number of tokens. As the gpt generations is set to 100 tokens max. Here we aim to also 100 tokens message. I am repeating it now. This is a filler message it same some extra text so as to help estimate the number of tokens. As the gpt generations is set to 100 tokens max. Here we aim to also 100 tokens message."

        elif "gpt" in engine_:
            assert not args.base_model_template

            if args.verbose:
                print_chat_messages(simulated_conv_messages)

            if args.azure_openai:
                # time.sleep(0.1)
                c = completions_with_backoff(
                    client=model,
                    model=openai_2_azure_tag[engine_],
                    messages=simulated_conv_messages,
                    max_tokens=100,
                    n=1,
                    temperature=1.0,
                )

            else:
                # todo: add backoff
                c = model.chat.completions.create(
                    model=engine_,
                    messages=simulated_conv_messages,
                    max_tokens=100,
                    n=1,
                    temperature=1.0,
                )
            response = c.choices[0].message.content

        elif "llama_2" in engine_:

            if args.base_model_template:
                assert args.system_message
                formatted_prompt, stop_words = apply_base_model_template(
                    simulated_conv_messages,
                    assistant_label=assistant_label,
                    user_label=user_label,
                    system_label=system_label,
                    add_generation_prompt=True,
                    return_stop_words=True
                )
                input_ids = tokenizer(formatted_prompt, return_tensors="pt").to(model.device).input_ids
                assert all([w.upper() in stop_words_up for w in stop_words])
                stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stop_words_up, tokenizer, input_ids)])

                print(f"\n>>>>>>>>>>>>FORMATTED<<<>>>PROMPT<<<<<<<<<<<<\n{formatted_prompt}\n>>>>>>>>>>><<<<<<<<<<<\n")

            else:
                input_ids = tokenizer.apply_chat_template(simulated_conv_messages, return_tensors="pt", add_generation_prompt=True).to(model.device)
                if args.verbose:
                    print_chat_messages(simulated_conv_messages)
                stopping_criteria = None

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
                    add_generation_prompt=True,
                    return_stop_words=True
                )
                input_ids = tokenizer(formatted_prompt, return_tensors="pt").to(model.device).input_ids
                assert all([w.upper() in stop_words_up for w in stop_words])
                stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stop_words_up, tokenizer, input_ids)])

                print(f"\n>>>>>>>>>>>>FORMATTED<<<>>>PROMPT<<<<<<<<<<<<\n{formatted_prompt}\n>>>>>>>>>>><<<<<<<<<<<\n")

            else:
                input_ids = tokenizer.apply_chat_template(simulated_conv_messages, return_tensors="pt", add_generation_prompt=True).to(model.device)
                if args.verbose:
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

            if args.base_model_template:
                assert args.system_message
                formatted_prompt, stop_words = apply_base_model_template(
                    simulated_conv_messages,
                    assistant_label=assistant_label,
                    user_label=user_label,
                    system_label=system_label,
                    add_generation_prompt=True,
                    return_stop_words=True
                )
                input_ids = tokenizer(formatted_prompt, return_tensors="pt").to(model.device).input_ids
                assert all([w.upper() in stop_words_up for w in stop_words])
                stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stop_words_up, tokenizer, input_ids)])

                if args.verbose:
                    print(f"\n>>>>>>>>>>>>FORMATTED<<<>>>PROMPT<<<<<<<<<<<<\n{formatted_prompt}\n>>>>>>>>>>><<<<<<<<<<<\n")

            else:
                input_ids = tokenizer.apply_chat_template(simulated_conv_messages, return_tensors="pt", add_generation_prompt=True).to(model.device)
                if args.verbose:
                    print_chat_messages(simulated_conv_messages)
                stopping_criteria = None

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
print("HF TOKEN:", hidden_token )


# choices = ["A", "B", "C", "D", "E", "F"]
choices = list(string.ascii_uppercase)

llama_dir = "/gpfswork/rech/imi/utu57ed/llama/llama_files/"

def get_prompt_skeleton(experiment_name, args, simulated_participant):

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

    if args.format == "chat":
        if args.query_prompt:
            query_str = args.query_prompt
        else:
            query_str = "Answer: ("

    else:
        raise ValueError(f"Undefined format {args.format}.")

    assert (not questionnaire_description_empty == questionnaire_description)

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


    # look for 'A.' -> change to A)
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

    else:
        raise ValueError(f"Undefined textual format {args.format}.")

    prompt = get_prompt_skeleton(
        experiment_name=experiment_name, args=args, simulated_participant=simulated_participant
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

    gpt_token_counter = {"input": 0, "output": 0}

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

        assert n_options == len(permutations_dict)
        answers = choices[:n_options]

        assert all([a in permutations_dict for a in answers])

        label = test_df.iloc[item_i, test_df.shape[1]-1]
        assert label in answers + ["undef"]

        if args.estimate_gpt_tokens:
            gpt_tokenizer = tiktoken.get_encoding("cl100k_base")
        else:
            gpt_tokenizer = None


        if args.simulate_conversation_theme:

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
                    llm_generator=llm_generator
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
                messages_conv=messages_conv if args.simulate_conversation_theme else None,
                add_query_str=True,  # not query_in_reply
            )
            n_input_tokens = sum([len(gpt_tokenizer.encode(msg['content'])) for msg in messages])

            gpt_token_counter['input'] += n_input_tokens
            gpt_token_counter['output'] += 1

        if engine == "dummy":

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
                add_generation_prompt=True,
                assistant_label=simulated_participant_to_name(
                    simulated_participant, args.simulated_population_type).upper(),
                user_label="USER",
                system_label="CONTEXT"
            )

            if args.verbose:
                print(f"************************\nFORMATTED PROMPT:\n{formatted_prompt}\n******************")

            # generation = messages[-2]['content'][messages[-2]['content'].index(") War") - 1:][:1]
            generation = random.choice([f"{c}" for c in answers])

            # import re
            # generation = messages[-2]['content'][messages[-2]['content'].index(") a few hours per day") - 1:][:1]

            # if re.search("\) You receive: 85", messages[-2]['content']):
            #     generation = messages[-2]['content'][messages[-2]['content'].index(") You receive: 85") - 1:][:1]
            # elif re.search("\) You receive: 100", messages[-2]['content']):
            #     generation = messages[-2]['content'][messages[-2]['content'].index(") You receive: 100") - 1:][:1]
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
                    add_generation_prompt=True,
                    assistant_label=simulated_participant_to_name(simulated_participant, args.simulated_population_type).upper(),
                    user_label="USER",
                    system_label="CONTEXT"
                )

            else:
                formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            if args.query_in_reply:
                formatted_prompt += f"{prompt['query_str']}"

            if args.verbose:
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
                    add_generation_prompt=True,
                    assistant_label=simulated_participant_to_name(simulated_participant, args.simulated_population_type).upper(),
                    user_label="USER",
                    system_label="CONTEXT"
                )

            else:
                formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            if args.query_in_reply:

                formatted_prompt += f"{prompt['query_str']}"

            if args.verbose:
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

        elif "gpt-3.5-turbo" in engine or "gpt-4" in engine:
            if args.query_in_reply:
                raise ValueError("Can't use query_in_reply with gpt models.")

            if args.base_model_template:
                raise ValueError("base_model_template not supported for gpt models")

            tokenizer, model = llm_generator

            messages = construct_messages(
                prompt=prompt,
                system_message=args.system_message,
                messages_conv=messages_conv if args.simulate_conversation_theme else None,
                add_query_str=True,
            )

            if args.verbose:
                print_chat_messages(messages)

            encoder = tiktoken.encoding_for_model(engine)

            # get the encoding for each letter in choices
            logit_bias = {encoder.encode(c)[0]: 100 for c in answers}

            if args.azure_openai:
                # time.sleep(0.05)
                c = completions_with_backoff(
                    client=model,
                    model=openai_2_azure_tag[engine],
                    messages=messages,
                    max_tokens=1,
                    n=1,
                    temperature=0,
                    logit_bias=logit_bias,
                )
            else:
                c = model.chat.completions.create(
                    model=engine,
                    messages=messages,
                    max_tokens=1,
                    n=1,
                    temperature=0,
                    logit_bias=logit_bias,
                )

            generation = c.choices[0].message.content

            lprobs = dummy_lprobs_from_generation(generation, answers, label_2_text_option_dict)

        else:
            raise ValueError(f"Not recognized model {engine}.")

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

    acc = np.mean(cors)
    cors = np.array(cors)
    all_scores = np.array(all_scores)

    if args.estimate_gpt_tokens:
        estimate_and_print_gpt_prices(gpt_token_counter, engine)

    return cors, acc, all_probs, all_lprobs, all_answers, all_scores, all_generations, gpt_token_counter


def main(args):
    engine = args.engine
    print("Engine:", engine)

    subjects = sorted([f.split("_test.csv")[0] for f in os.listdir(os.path.join(args.data_dir, "test")) if "_test.csv" in f])

    # dump results dir
    dump_results_dir = os.path.join(args.save_dir, "_".join([
        args.experiment_name,
        engine,
        os.path.basename(args.data_dir),
        f"permutations_{args.permutations}" if args.permutations > 1 else "",
        f"permute_options_{args.permute_options_seed}" if args.permute_options else "",
        f"format_{args.format}",
        f"simulate_conv_{args.simulate_conversation_theme}" if args.simulate_conversation_theme else "",
        timestamp
    ]))

    if not args.overwrite:
        # check for previous versions and break if found
        import glob
        prev_versions = glob.glob(dump_results_dir.removesuffix(timestamp)+"*/results.json")
        if len(prev_versions) > 0:
            raise RuntimeError(f"Previous version of this run were found: {prev_versions}")

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
        tokenizer = AutoTokenizer.from_pretrained(f"microsoft/{engine}", trust_remote_code=True, cache_dir=hf_cache_dir)
        model = AutoModelForCausalLM.from_pretrained(f"microsoft/{engine}", trust_remote_code=True, cache_dir=hf_cache_dir, device_map="cuda")
        llm_generator = (tokenizer, model)

    elif "Mistral-7B" in engine and "_ft_" in engine:

        ft_model_path = f"./results_ft/{engine}/final"

        if "no_peft" in engine:
            model = AutoModelForCausalLM.from_pretrained(
                ft_model_path,
                device_map="auto",
                cache_dir=hf_cache_dir
            )
            tokenizer = AutoTokenizer.from_pretrained(ft_model_path, cache_dir=hf_cache_dir)

        else:

            lora_config = LoraConfig.from_pretrained(ft_model_path)
            base_model = lora_config.base_model_name_or_path

            if "LOAD_INSTRUCT" in ft_model_path:
                colored("LOADING INSTRUCT MODEL WHICH is different from the trained based model... Only for testing!!", "red")
                base_model = "mistralai/Mistral-7B-Instruct-v0.2"

            print(f"Loading {engine}")

            bnb_config_ = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=False,
            )
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                quantization_config=bnb_config_,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
                cache_dir=hf_cache_dir
            )
            print("Loaded base model: ", base_model)
            model = prepare_model_for_kbit_training(model)
            model = PeftModel.from_pretrained(model, ft_model_path)
            print("Loaded peft from ", ft_model_path)


            try:
                tokenizer = AutoTokenizer.from_pretrained(ft_model_path, trust_remote_code=True, cache_dir=hf_cache_dir)
                print("Loaded tokeniezer from ", ft_model_path)
            except:
                tokenizer = AutoTokenizer.from_pretrained(
                    lora_config.base_model_name_or_path, trust_remote_code=True, cache_dir=hf_cache_dir)
                print("Loaded tokenizer from ", lora_config.base_model_name_or_path)

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
        model_name = engine.rstrip("-4b")
        print(f"Loading {engine} -> {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(f"mistralai/{model_name}", cache_dir=hf_cache_dir, device_map="auto")
        # model = AutoModelForCausalLM.from_pretrained(f"mistralai/{model_name}", device_map="auto", cache_dir=hf_cache_dir, load_in_4bit=True, attn_implementation="flash_attention_2")
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

    elif "gpt" in engine:

        if args.azure_openai:
            print(colored("Using Azure OPENAI API", "red"))
            from openai import AzureOpenAI

            if engine == "gpt-3.5-turbo-0125":
                model = AzureOpenAI(
                    azure_endpoint="https://petunia-grgur.openai.azure.com/",
                    api_key=os.getenv("AZURE_OPENAI_KEY_gpt_35_turbo_0125"),
                    api_version="2024-02-15-preview"
                )

            elif engine == "gpt-3.5-turbo-1106":
                model = AzureOpenAI(
                    azure_endpoint="https://petunia-grgur-gpt-35-turbo-1106.openai.azure.com/",
                    api_key=os.getenv("AZURE_OPENAI_KEY_gpt_35_turbo_1106"),
                    api_version="2024-02-15-preview"
                )
            else:
                raise NotImplementedError("Azure endpoint not found.")

        else:
            print(colored("Using OPENAI API", "red"))
            from openai import OpenAI
            openai_api_key = os.environ["OPENAI_API_KEY"]
            hidden_key = openai_api_key[:4] + "*" * 10 + openai_api_key[4:]
            print(f"OPENAI KEY: {hidden_key}")
            model = OpenAI(api_key=openai_api_key)

        tokenizer = tiktoken.get_encoding("cl100k_base")
        llm_generator = (tokenizer, model)

    elif engine or engine in ["dummy", "interactive"]:
        llm_generator = None

    else:
        raise ValueError(f"Undefined model: {engine}")

    print(f"Loaded model: {args.engine}.")

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

        # evaluate over population
        for sim_part_i, (simulated_participant, simulated_participant_gender, participant_perm_dicts) in enumerate(zip(simulated_population, simulated_population_genders, permutations_dicts)):
            print(f"Simulated participant {sim_part_i}/{len(simulated_population)}")

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
            gpt_tokens_total['input'] += gpt_tokens['input']
            gpt_tokens_total['output'] += gpt_tokens['output']

            subj_acc[sim_part_i][subject] = acc
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
    parser.add_argument("--query-in-reply", action="store_true", help="Force the query string as the beginning of the model's reply.")
    parser.add_argument("--base-model-template", action="store_true")
    parser.add_argument("--query-prompt", "-qp", type=str, help="Use Answer(as ONE letter): where applicable.")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--system-message", "-sm", action="store_true")
    parser.add_argument("--assert-params", action="store_true")
    parser.add_argument("--cold-run", "-cr", action="store_true")
    parser.add_argument("--estimate-gpt-tokens", "-t", action="store_true")
    parser.add_argument("--eval-set", type=str, default="test", choices=["test", "val"])
    parser.add_argument("--simulate-conversation-theme", type=str, default=None)
    parser.add_argument("--simulated-conversation-n-messages", type=int, default=5)
    parser.add_argument("--permute-options", "-po", action="store_true")
    parser.add_argument("--azure-openai", action="store_true")
    parser.add_argument("--simulated-human-knows-persona", action="store_true")
    parser.add_argument("--simulated-population-type", "-pop", type=str, default="tolkien_characters", choices=["permutations", "tolkien_characters", "famous_people", "llm_personas", "user_personas", "anes"])
    parser.add_argument("--permutations", "-p", type=int, default=1)  # permutations as a population type
    parser.add_argument("--permute-options-seed", type=str)
    parser.add_argument("--separator", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    assert args.azure_openai
    assert args.pvq_version == "pvq_auto"
    assert args.format == "chat"

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

            if "Instruct" in args.engine or "ft_roleplay" in args.engine:
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
        print("System message:", args.system_message)
        # just used to show the profile to be used
        exit()

    if ("gpt-3.5" in args.engine and args.permutations > 50) or ("gpt-4" in args.engine and args.permutations > 5):
        raise ValueError(f"Are you sure you want to use {args.permutations} with {args.engine}??")

    # assert for plosone gpt or query_in_reply for other models
    # because query_in_reply can't be implemented for GPTs
    assert args.query_in_reply or "gpt" in args.engine or "dummy" in args.engine

    if "gpt" in args.engine:
        if args.query_in_reply:
            raise ValueError("Can't use query in reply with gpt models")

    start_time = time.time()
    main(args)
    end_time = time.time()
    print("Elapsed time:", str(datetime.timedelta(seconds=end_time-start_time)).split(".")[0])


