import argparse
import datetime
import random
import re
import json

import matplotlib.pyplot as plt
import tiktoken


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

choices = ["A", "B", "C", "D", "E", "F"]
# choices = ["a", "b", "c", "d", "e", "f"]
# choices = ["1", "2", "3", "4", "5", "6"]

llama_dir = "/gpfswork/rech/imi/utu57ed/llama/llama_files/"

def get_prompts_skeleton(subject, experiment_name, profile_dict):

    if "hofstede" in experiment_name:
        if args.direct_perspective:
            raise NotImplementedError("Profile not implemented.")


        # default
        prompts = {
            "intro": "The following is a questionnaire (with answers)\n\n",
            "query": f"\nAnswer:",
        }
        if profile_dict:

            # we define the perspective through a profile
            profile_string = "\n".join([
                f"\t{k}:{v}" for k, v in profile_dict.items()
            ])

            prompts = {
                "intro": "The following is a questionnaire (with answers) given to a person with the following profile:\n"+
                         f"{profile_string}\n\n",
                "query": f"\nAnswer (from the person):",
            }

    elif "pvq" in experiment_name:

        questionniare_description = "Here we briefly describe some people. Please read each description and think about how much each person is or is not like you. Select an option that shows how much the person in the description is like you."
        # questionniare_description = "Here we briefly describe some people. Please read each description and think about how much each person is or is not like you. Put an X in the box to the right that shows how much the person in the description is like you.",

        # default
        prompts = {
            "intro": questionniare_description + "\n\n",
            "query": f"\nAnswer:",
        }

        if not profile_dict:
            if args.direct_perspective:
                raise NotImplementedError("Profile not implemented.")

        if "hitler" in experiment_name:
            prompts = {
                "intro": "The following is a questionnaire (with answers) given to Adolf Hitler.\n\n"+
                         questionniare_description + "\n\n",
                "query": f"\nAnswer (from Adolf Hitler):",
            }

        elif "gandhi" in experiment_name:
            prompts = {
                "intro": "The following is a questionnaire (with answers) given to Mahatma Gandhi.\n\n"+
                         questionniare_description + "\n\n",
                "query": f"\nAnswer (from Mahatma Gandhi):",
            }

        elif "dorian" in experiment_name:
            prompts = {
                "intro": "The following is a questionnaire (with answers) given to Dorian Gray.\n\n"+
                         questionniare_description + "\n\n",
                "query": f"\nAnswer (from Dorian Gray):",
            }

        elif profile_dict:
            # we define the perspective through a profile
            profile_string = "\n".join([
                f"\t{k}:{v}" for k, v in profile_dict.items()
            ])

            if args.direct_perspective:
                set_profile_str = "You are a person with the following profile:"
                query_str = "Answer:"

            else:
                set_profile_str = "The following is a questionnaire (with answers) given to a person with the following profile:"
                query_str = "Answer (from the person):"

            prompts = {
                "intro": f"{set_profile_str}\n"+
                         f"{profile_string}\n\n"+
                         f"{questionniare_description}\n\n",
                "query": f"\n{query_str}",
            }

    elif "mmlu" in experiment_name:

        questionniare_description = "The following are multiple choice questions (with answers)."
        prompts = {
            "intro": f"{questionniare_description}\n\n",
            "query": f"\nAnswer:"
        }

        if profile_dict:
            # we define the perspective through a profile
            profile_string = "\n".join([
                f"\t{k}:{v}" for k, v in profile_dict.items()
            ])

            if args.direct_perspective:
                set_profile_str = "You are a person with the following profile:"
                query_str = "Answer:"

            else:
                set_profile_str = "The following is a questionnaire (with answers) given to a person with the following profile:"
                query_str = "Answer (from the person):"

            prompts = {
                "intro": f"{set_profile_str}\n"+
                         f"{profile_string}\n\n",
                "query": f"\n{query_str}",
            }


    elif subject == "political_compass":
        if profile_dict:
           raise NotImplementedError("Profile not implemented.")

        if args.direct_perspective:
            raise NotImplementedError("Profile not implemented.")
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

        prompts = {
            "intro": "",
            "query": "\nAnswer:"
        }

        if args.direct_perspective:
            raise NotImplementedError("Profile not implemented.")

        if profile_dict:
            # we define the perspective through a profile
            profile_string = "\n".join([
                f"\t{k}:{v}" for k, v in profile_dict.items()
            ])

            prompts = {
                "intro": "The following is a questionnaire (with answers) given to a person with the following profile:\n"+
                         f"{profile_string}\n\n",
                "query": f"\nAnswer (from the person):",
            }

        # if "tomi_default" in experiment_name:
        #     # in-context examples are not from the same age as the query age
        #     prompts = {
        #         "intro": "",
        #         "query": "\nAnswer:"
        #     }
        #
        # if "tomi_age" in experiment_name:
        #     numbers = [int(num) for num in re.findall(r'\d+', experiment_name)]
        #     age = ["3", "5", "10", "15", "30", "50"][numbers[0]]
        #
        #     # in-context examples are not from the same age as the query age
        #     prompts = {
        #         # "intro": "The following is a text comprehension test with answers from a {} year old.\n\n".format(age),
        #         "intro": "The following is a text comprehension test with answers from people of various ages.\n\n".format(age),
        #         "query": "\nAnswer (from a {} year old):".format(age)
        #     }
        #
        # if "tomi_default_new" in experiment_name:
        #     # in-context examples are not from the same age as the query age
        #     prompts = {
        #         "intro": "The following are multiple choice questions (with answers).\n\n",
        #         "query": "\nAnswer:"
        #     }
        #
        # if "tomi_age_new" in experiment_name:
        #     numbers = [int(num) for num in re.findall(r'\d+', experiment_name)]
        #     age = ["3", "5", "10", "15", "30", "50"][numbers[0]]
        #
        #     # in-context examples are not from the same age as the query age
        #     prompts = {
        #         "intro": "The following are multiple choice questions (with answers from {} year old human).\n\n".format(age),
        #         "query": "\nAnswer (from a {} year old):".format(age),
        #     }
        #
        # if "tomi_dog" in experiment_name:
        #     # in-context examples are not from the same age as the query age
        #     prompts = {
        #         "intro": "The following is a text comprehension test with answers from a dog.\n\n",
        #         "query": "\nAnswer (from a dog):"
        #     }

    else:
        if profile_dict:
            raise NotImplementedError("Profile not implemented.")

        if args.direct_perspective:
            raise NotImplementedError("Profile not implemented.")

        # default
        prompts = {
            "intro": "The following are multiple choice questions (with answers) about {}.\n\n".format(
                format_subject(subject)),
            "query": "\nAnswer:"
        }

        if experiment_name == "spec_prof":
            prompts = {
                "intro": "The following is an interview with a professor of {}.\n\n".format(format_subject(subject)),
                "query": "\nAnswer from the professor:"
            }

        if experiment_name == "unspec_prof":
            prompts = {
                "intro": "The following is an interview with a professor.\n\n",
                "query": "\nAnswer from the professor:"
            }

    return prompts

def dummy_lprobs_from_generation(response):
    # lprobs (todo: this is hardcoded)
    lprobs = [-100] * len(choices)
    for i, op in enumerate(choices):
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


def format_example(df, idx, subject, experiment_name, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2

    num_options = 0
    for j in range(k):
        op_str = df.iloc[idx, j+1]

        if op_str == "undef":
            continue

        prompt += "\n{}. {}".format(choices[j], op_str)
        num_options += 1

    # makes in-context examples be from a 50-year-old
    if "tomi_age" in experiment_name and include_answer:

        # include_answer is true -> this is an in-context example
        # create dummy examples -> change the age to 50, so that the correct answers make sense

        query_str = get_prompts_skeleton(subject=subject, experiment_name=experiment_name, profile_dict=args.profile_dict)["query"]
        assert "year old" in query_str
        new_query_str = re.sub(r'\d+', '50', query_str)

        prompt += new_query_str

    else:
        # add query prompt (ex. "\nAnswer:")
        prompt += get_prompts_skeleton(subject=subject, experiment_name=experiment_name, profile_dict=args.profile_dict)["query"]

    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt, num_options

def gen_prompt(train_df, subject, experiment_name, k=-1):

    # get intro prompt (ex. "The following are .... \n\n" )
    prompt = get_prompts_skeleton(subject=subject, experiment_name=experiment_name, profile_dict=args.profile_dict)["intro"]

    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        example_prompt, _ = format_example(train_df, i, subject=subject, experiment_name=experiment_name)
        prompt += example_prompt
    return prompt


def eval(args, subject, engine, dev_df, test_df, llama_generator=None):
    cors = []
    all_probs = []
    all_answers = []
    answers = choices[:test_df.shape[1]-2]

    gpt_token_counter = 0

    for i in range(test_df.shape[0]):
        if i % 10 == 0:
            print(f"Eval progress: {i}/{test_df.shape[0]}")

        # get prompt and make sure it fits
        k = args.ntrain
        prompt_end, n_options = format_example(test_df, i, subject=subject, experiment_name=args.experiment_name, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, experiment_name=args.experiment_name, k=k)
        prompt = train_prompt + prompt_end

        #  crop to 2048 tokens
        #  this is used when the prompt is too long it feeds the most possible number of examples that fit
        while crop(prompt) != prompt:
            k -= 1
            train_prompt = gen_prompt(dev_df, subject, k)
            prompt = train_prompt + prompt_end

        label = test_df.iloc[i, test_df.shape[1]-1]
        assert label in choices + ["undef"]

        if args.estimate_gpt_tokens:
            encoder = tiktoken.encoding_for_model('gpt-3.5-turbo-0301')
            assert encoder == tiktoken.encoding_for_model('gpt-4-0314')
            gpt_token_counter += len(encoder.encode(prompt)) + 1  # prompt + 1 generated token

        if engine == "dummy":
            generation = random.choice([f"{c} ba" for c in choices])
            lprobs = dummy_lprobs_from_generation(generation)

        elif engine == "interactive":
            # ask the user to choose
            generation = input(f"{prompt}")
            lprobs = dummy_lprobs_from_generation(generation)

        elif engine in ["llama_7B", "llama_13B", "llama_30B", "llama_65B"]:
            if args.generative_qa:
                result = llama_generator.generate(
                    [prompt],
                    max_gen_len=5,
                    temperature=0,
                )[0]

                generation = remove_prefix(result, prompt)
                lprobs = dummy_lprobs_from_generation(generation)

            else:
                results, top_logprobs = llama_generator.generate_next_token(
                    [prompt],
                    max_gen_len=1,
                    temperature=0,
                    logprobs=100
                )

                # result = results[0]
                top_logprobs = top_logprobs[0]

                lprobs = []
                for ans in answers:
                    print(f"ans {ans} : {top_logprobs.get(ans, -100)}")
                    lprobs.append(top_logprobs.get(ans, -100))

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
                                encoder.encode(f" {c}")[0]: 100 for c in choices[:n_options]
                            }

                        else:
                            logit_bias = {
                                encoder.encode(c)[0]: 100 for c in choices[:n_options]
                            }


                    if args.system_message:
                        assert prompt == train_prompt+prompt_end

                        # user message
                        c = openai.ChatCompletion.create(
                            model=engine,
                            messages=[
                                {"role": "system", "content": train_prompt},
                                # {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
                                {"role": "user", "content": prompt_end}
                            ],
                            max_tokens=max_tokens,
                            n=1,
                            temperature=0,
                            logit_bias=logit_bias,
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
                        )

                    break
                except Exception as e:
                    print(e)
                    print("Pausing")
                    time.sleep(10)
                    continue

            generation = c['choices'][0]['message']['content']

            if args.generative_qa:
                if generation not in choices:
                    raise ValueError("Generation is not in choices and gqa is not used. Potential problem with logit bias?")


            lprobs = dummy_lprobs_from_generation(generation)

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

        else:
            raise ValueError(f"Not recotnized model {engine}.")

        if args.verbose:
            if args.system_message:
                print(f"Prompt(System):\n{train_prompt}")
                print(f"Prompt(User):\n{prompt}")

            else:
                print(f"Prompt:\n{prompt}")

        probs = softmax(np.array(lprobs))

        if args.generative_qa:

            first_generated_letter = generation.strip()[:1]
            if first_generated_letter in choices:
                pred = first_generated_letter
            else:
                pred = "other"


            # whitespace before label is ok
            cor = generation.strip().startswith(label)

        else:
            pred = {
                i: c for i, c in enumerate(choices)
            }[np.argmax(lprobs)]
            cor = pred == label

        if args.verbose:
            print(f"Pred:{pred} (Generation:{generation})")

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
        dump_results_dir = os.path.join(args.save_dir, "_".join(
            [args.experiment_name, engine, args.data_dir, f"ntrain_{args.ntrain}_" + f"profile_{args.profile}" if args.profile else "", timestamp]))
        os.makedirs(dump_results_dir, exist_ok=True)

        if engine in ["llama_7B", "llama_13B", "llama_30B", "llama_65B"]:
            # load llama
            from example_llama import setup_model_parallel, load
            local_rank, world_size = setup_model_parallel()

            if local_rank > 0:
                sys.stdout = open(os.devnull, 'w')

            model_size = remove_prefix(engine, "llama_")
            assert model_size in ["7B", "13B", "30B", "65B"]

            llama_ckpt_dir = os.path.join(llama_dir, model_size)
            llama_tokenizer_path = os.path.join(llama_dir, "tokenizer.model")

            # load model
            llama_generator = load(llama_ckpt_dir, llama_tokenizer_path, local_rank, world_size)

        else:
            llama_generator = None

        all_cors = []

        subj_acc = {}
        subj_len = {}

        metrics = {}
        answers = {}

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

            # print("Example of prompt:\n")
            # print(get_prompts_skeleton(subject, experiment_name=args.experiment_name)["intro"])

            cors, acc, probs, preds, gpt_tokens = eval(args, subject, engine, dev_df, test_df, llama_generator=llama_generator)
            all_cors.append(cors)
            gpt_tokens_total += gpt_tokens

            subj_acc[subject] = acc
            subj_len[subject] = len(test_df)
            answers[subject] = preds

            def map_choice_to_number(letter):
                # A-F -> 1-6
                # find index of letter in choices and add 1
                number = choices.index(letter) + 1

                if letter in "ABCDEF":
                    assert number == ord(letter) - ord('A') + 1

                return number

            if "hofstede" in args.data_dir:
                assert "hofstede" in args.experiment_name

                preds_values = np.vectorize(map_choice_to_number)(preds)

                # from the manual (question indices start from 1)
                # power_distance = 35(m07 – m02) + 25(m20 – m23) + C(pd)
                # individualism = 35(m04 – m01) + 35(m09 – m06) + C(ic)
                # masculinity = 35(m05 – m03) + 35(m08 – m10) + C(mf)
                # uncertainty_avoidance = 40(m18 - m15) + 25(m21 – m24) + C(ua)
                # long_term_orientation = 40(m13 – m14) + 25(m19 – m22) + C(ls)
                # indulgence = 35(m12 – m11) + 40(m17 – m16) + C(ir)

                # indices start from 0
                metrics[subject] = {
                    "power_distance": 35*(preds_values[6] - preds_values[1]) + 25*(preds_values[19] - preds_values[22]),
                    "individualism": 35*(preds_values[3] - preds_values[0]) + 35*(preds_values[8] - preds_values[5]),
                    "masculinity": 35*(preds_values[4] - preds_values[2]) + 35*(preds_values[7] - preds_values[9]),
                    "uncertainty_avoidance": 40*(preds_values[17] - preds_values[14]) + 25*(preds_values[20] - preds_values[23]),
                    "long_term_orientation": 40*(preds_values[12] - preds_values[13]) + 25*(preds_values[18] - preds_values[21]),
                    "indulgence": 35*(preds_values[11] - preds_values[10]) + 40*(preds_values[16] - preds_values[15])
                }
                metrics[subject] = {k: float(v) for k, v in metrics[subject].items()}

            elif "pvq" in args.data_dir:
                assert "pvq" in args.experiment_name

                # pvq is evaluated by averaging scored based on different values
                preds_values = np.vectorize(map_choice_to_number)(preds)

                profile_values_idx_json = os.path.join(os.path.join(args.data_dir, "raw"), "values.json")
                with open(profile_values_idx_json) as f:
                    profile_values_idx = json.load(f)
                profile_values_idx = {k: np.array(v)-1 for k, v in profile_values_idx.items() if k != "_comment"}

                metrics[subject] = {}

                # mean_values = preds_values.mean()

                for profile_value, idxs in profile_values_idx.items():
                    # metrics[subject][profile_value] = preds_values[idxs].mean() - mean_values
                    metrics[subject][profile_value] = preds_values[idxs].mean()

            elif "political_compass" in args.data_dir:
                # political compas is evaluated using the website
                resp_df = test_df.copy()
                resp_df[5] = preds
                preds_csv_file = f"./results/political_compass/preds_{args.experiment_name}_{engine}_{timestamp}.csv"
                resp_df.to_csv(preds_csv_file, header=None, index=False)
                print(f"preds saved to '{preds_csv_file}")

                evaluate_csv_file(preds_csv_file)

            else:
                metrics[subject] = {
                    "accuracy": subj_acc[subject]
                }

            test_df["{}_correct".format(engine)] = cors
            for j in range(probs.shape[1]):
                choice = choices[j]
                test_df["{}_choice{}_probs".format(engine, choice)] = probs[:, j]

            if args.log:
                test_df.to_csv(os.path.join(args.save_dir, "results_{}".format(engine), "{}.csv".format(subject)), index=None)

        weighted_acc = np.mean(np.concatenate(all_cors))

        for subj, m in metrics.items():
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
                **subj_acc,
                **{
                    "average": weighted_acc
                },
                "metrics": metrics,
                "answers": answers,
                **{
                    "params": vars(args)
                }
            }, fp, indent=4)

        print(f"Results saved to {json_dump_path}")

        print("")
        print("Average accuracy per subject.")
        for subject in subjects:
            print("{} accuracy ({}): {:.3f}".format(subject, subj_len[subject], subj_acc[subject]))

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
    ], default=["davinci", "curie", "babbage", "ada"], nargs="+")
    parser.add_argument('--profile', type=str, help='Profile definition in format "k:v;k:v;k:v", ex. "age:35;interests:reading books"')
    parser.add_argument("--generative_qa", "-gqa", action="store_true", help="Use generative question answering instead of MCQ.")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--system-message", "-sm", action="store_true")
    parser.add_argument("--direct-perspective", action="store_true")
    parser.add_argument("--cold-run", "-cr", action="store_true")
    parser.add_argument("--estimate-gpt-tokens", "-t", action="store_true")
    parser.add_argument("--match-tokens-with-space", action="store_true")
    parser.add_argument("--eval-set", type=str, default="test", choices=["test","val"])

    parser.add_argument("--log", "-l", type=bool, default=False)  # doesn't work well for multiproc (bigger llama models) # remove this parameter?
    args = parser.parse_args()

    profile = {}
    if args.profile:
        for item in args.profile.split(';'):
            key, value = item.split(':')
            profile[key] = value

        args.profile_dict = profile
    else:
        args.profile_dict = None

    print(f"Profile:\n{profile}")
    if args.cold_run:
        # just used to show the profile to be used
        exit()


    main(args)

