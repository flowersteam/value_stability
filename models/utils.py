from termcolor import colored

def print_chat_messages(messages):
    print("*********************")
    for msg in messages:
        role = msg['role'].upper()
        color = {
            "SYSTEM": "red",
            "USER": "blue",
            "ASSISTANT": "green",
        }.get(role, None)
        print(f"{colored(role + ':', color)}{msg['content']}")
    print("*********************")

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

def construct_messages(prompt, system_message, messages_conv=None):

    set_persona_str = prompt["set_persona_str"]
    questionnaire_description = prompt["questionnaire_description"]

    user_prompt = f"{questionnaire_description}\n\n" if questionnaire_description else ""
    user_prompt += prompt["item_str"]

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



def secs_2_hms(s):
    minutes, seconds = divmod(s, 60)
    hours, minutes = divmod(minutes, 60)
    return hours, minutes, seconds

