import copy
import json
import time

from termcolor import cprint
import numpy as np
from utils import softmax, create_choices_str, map_choice_to_number


def create_response_msg(query_str, pred, chosen_value):
    # add responses to the conversation history
    response_string = create_choices_str([pred], [chosen_value]).lstrip()

    assert response_string[0] == "("
    response_string = response_string.lstrip("(")

    resp_msg_assistant = query_str + response_string

    return {"role": "assistant", "content": resp_msg_assistant}


def score_non_extreme_value_svs(
        test_df,
        item_i,
        format_example,
        args,
        permutations_dict,
        llm_generator,
        first_non_extreme_value,
        construct_messages,
        previous_messages,
        answers,
        label_2_text_option_dict,
        simulated_participant,
):
    score_prompt, most_n_options = format_example(
        test_df, item_i,
        experiment_name=args.experiment_name,
        args=args,
        permutations_dict=permutations_dict,
        simulated_participant_description=simulated_participant['description'],
        base_model_template=llm_generator.base_model_template
    )

    score_prompt['set_persona_str'] = ""

    if first_non_extreme_value:
        score_prompt['questionnaire_description'] = "Now, rate the rest of the values."
    else:
        score_prompt['questionnaire_description'] = ""

    # Add new item to messages
    messages = construct_messages(
        prompt=score_prompt,
        system_message=llm_generator.system_message,
        messages_conv=previous_messages
    )

    cprint("\nQuery the score of a non extreme value", "green")
    generation, lprobs = llm_generator.predict(
        messages=messages,
        answers=answers,
        label_2_text_option_dict=label_2_text_option_dict,
        query_string=score_prompt['query_str'],
        assistant_label=simulated_participant["name"].upper()
    )

    probs = softmax(np.array(lprobs))
    pred = {i: c for i, c in enumerate(answers)}[np.argmax(lprobs)]
    score = map_choice_to_number(pred, permutations_dict, offset=-1)
    print("Result:")
    print(f"\t-generation: {generation}")
    print(f"\t-pred: {pred}")
    print(f"\t-chosen value score: {score}")

    # add responses to the conversation history
    # add query and response to messages
    resp_assistant_msg = create_response_msg(score_prompt['query_str'], pred, score)
    messages.append(resp_assistant_msg)

    cor = False

    return cor, lprobs, probs, pred, generation, score, messages


svs_item_str_prefix_dict = {
    # group -> most/least important -> str
    1: {
        "most": f"Before you begin, read the values in List I, reply with a letter denoting the one that is most important to you and rate its importance.",
        "least": f"Now, reply with a letter denoting the value that is most opposed to your values and rate it -1.  If there is no such value, choose the value least important to you and rate it 0 or 1, according to its importance."
    },
    2: {
        "most": f"Before you begin, read the values in List II, reply with a letter denoting the one that is the most important to you and rate its importance.",
        "least": f"Now, reply with a letter denoting the value that is most opposed to your values, or - if there is no such value - choose the value least important to you, and rate it -1, 0, or 1, according to its importance."
    }

}

svs_group_description_dict = {
    1: "VALUES LIST I\n",
    2: "VALUES LIST II\n\nNow rate how important each of the following values is for you as a guiding principle in YOUR life.  These values are phrased as ways of acting that may be more or less important for you.  Once again, try to distinguish as much as possible between the values by using all the numbers.\n"
}

# group 1 starts at the 0-th item
# group 2 starts at the 30-th item
first_index_to_svs_group = {0: 1, 30: 2}
svs_groups_start_indices = list(first_index_to_svs_group.keys())
svs_group_to_size = {1: 30, 2: 27}


def get_values_for_group(group_id):

    if group_id == 1:
        with open(f'data/data_svs/raw/values_list_1.json', 'r') as file:
            group_values = json.load(file)['values']

    elif group_id == 2:
        with open(f'data/data_svs/raw/values_list_2.json', 'r') as file:
            group_values = json.load(file)['values']
    else:
        raise ValueError(f"Unknown group_id: {group_id}")

    return group_values


def choose_extreme_value(
        group_id,
        group_values,
        extreme_value_str,
        create_choices_str,
        choices,
        group_values_to_choose_from,
        prompt,
        construct_messages,
        llm_generator,
        previous_messages,
        label_2_text_option_dict,
        simulated_participant,
):

    item_str_prefix = svs_item_str_prefix_dict[group_id][extreme_value_str]

    # add options
    choice_str = create_choices_str(choices, group_values_to_choose_from)
    item_str = item_str_prefix + f"\n{choice_str}"

    select_extreme_value_prompt = copy.copy(prompt)

    # add group description
    if extreme_value_str == "most":
        group_description = svs_group_description_dict[group_id]
        select_extreme_value_prompt['questionnaire_description'] += f"\n\n{group_description}"
    else:
        select_extreme_value_prompt['questionnaire_description'] = ""
        select_extreme_value_prompt['set_persona_str'] = ""

    select_extreme_value_prompt['item_str'] = item_str

    # create the conversation history and query the model
    messages = construct_messages(
        prompt=select_extreme_value_prompt,
        system_message=llm_generator.system_message,
        messages_conv=previous_messages
    )

    cprint(f"\nGROUP {group_id}  Extreme value str: {extreme_value_str}", "green")
    group_answers = choices[:len(group_values_to_choose_from)]
    generation, lprobs = llm_generator.predict(
        messages=messages,
        answers=group_answers,
        label_2_text_option_dict=label_2_text_option_dict,
        query_string=prompt['query_str'],
        assistant_label=simulated_participant["name"].upper()
    )
    chosen_i = np.argmax(lprobs)
    pred = {i: c for i, c in enumerate(group_answers)}[chosen_i]
    chosen_value = group_values_to_choose_from[chosen_i]

    item_i_in_group = group_values.index(chosen_value)

    print("Result")
    print(f"\t-generation: {generation}")
    print(f"\t-pred: {pred}")
    print(f"\t-chosen {extreme_value_str} important value: {chosen_value}")

    # add query and response to messages
    resp_assistant_msg = create_response_msg(prompt['query_str'], pred, chosen_value)
    messages.append(resp_assistant_msg)

    return chosen_value, item_i_in_group, messages


def score_extreme_value(
        format_example,
        test_df,
        chosen_item_i,
        args,
        participant_perm_dicts,
        simulated_participant,
        llm_generator,
        construct_messages,
        previous_messages,
        answers,
        label_2_text_option_dict,
        prompt,
        chosen_value,
):
    score_prompt, most_n_options = format_example(
        test_df, chosen_item_i,
        experiment_name=args.experiment_name,
        args=args,
        permutations_dict=participant_perm_dicts[chosen_item_i],
        simulated_participant_description=simulated_participant["description"],
        base_model_template=llm_generator.base_model_template
    )

    score_prompt["questionnaire_description"] = f"You chose {chosen_value}, now rate its importance."
    score_prompt['set_persona_str'] = ""

    messages = construct_messages(
        prompt=score_prompt,
        system_message=llm_generator.system_message,
        messages_conv=previous_messages
    )

    # query the model
    print("\nQuery the score")
    generation, lprobs = llm_generator.predict(
        messages=messages,
        answers=answers,
        label_2_text_option_dict=label_2_text_option_dict,
        query_string=prompt['query_str'],
        assistant_label=simulated_participant["name"].upper()
    )

    probs = softmax(np.array(lprobs))
    pred = {i: c for i, c in enumerate(answers)}[np.argmax(lprobs)]
    cor = False
    score = map_choice_to_number(pred, participant_perm_dicts[chosen_item_i], offset=-1)

    print("Result:")
    print(f"\t-generation: {generation}")
    print(f"\t-pred: {pred}")
    print(f"\t-chosen score: {score}")

    # add query and response to messages
    resp_assistant_msg = create_response_msg(prompt['query_str'], pred, score)
    messages.append(resp_assistant_msg)

    return cor, lprobs, probs, pred, generation, score, messages
