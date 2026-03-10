import json
from pathlib import Path
import numpy as np

from models.utils import fix_alternating_msg_order, print_chat_messages

from personas.utils import simulated_participant_to_name

# taken from: https://github.com/numpy/numpy/blob/v1.26.0/numpy/linalg/umath_linalg.cpp
with open('contexts/code.txt', 'r') as f: code_str = f.read()

# taken from : https://weather.tomorrow.io/
with open('contexts/weather.txt', 'r') as f: weather_str = f.read()

# taken from: https://www.reddit.com/r/bjj/comments/e0ycek/a_beginners_guide_to_bjj_please_read_me_white/
with open('contexts/bjj.txt', 'r') as f: bjj_str = f.read()

# taken from: https://www.reddit.com/r/bjj/comments/e0ycek/a_beginners_guide_to_bjj_please_read_me_white/
with open('contexts/britney.txt', 'r') as f: britney_str = f.read()

# take the theme starter
opening_questions_for_themes = {
    "poem": "Hello, let's write a poem together. You start by the first verse I'll add the second one, and so on.",
    "joke": "Tell me a joke.",
    "history": "What is the significance of the battle of Hastings. Answer in two sentences.",
    "chess": "1. e4",
    "grammar": "Can you check this sentence for grammar? \n Whilst Jane was waiting to meet hers friend their nose started bleeding.",

    "bicycle": "Explain in two sentences what is the difference between a freewheel and a cassette on a bicycle?",
    "santa": "Ho, ho, ho! I am Santa Claus. What would you like for Christmas?",
    "chord": "Can you perform a slide transformation on C#m by sliding down the third note? What notes does that give us?",
    "code": f"What does this do? \n{code_str}",
    "year": "How do you picture a year in your mind?",  # taken from THRED dataset TEST set

    # reddit post: https://www.reddit.com/r/AskCulinary/comments/1cjrbtf/swedish_meatballs_cant_be_made_with_sweet_sour/
    "cooking": """I work in an industrial kitchen and my boss insists that Swedish Meatballs is a dish made by adding sweet n sour Asian sauce to meatballs. I have shown him many pictures but he says the only reason it is a white gravy instead of clear reddish is because they add flour to the sweet n sour. I have shown him many recipes but he says I'm lying and confused basically. Is there some other dish that IS meatballs cooked in sweet n sour asian sauce? I'm not talking "adding some vinegar" I'm talking my boss told me to cook this dish with 1Gal of premade Sysco Sweet n Sour and I've been argueing for 4 days that that's the grossest thing I've ever heard of. I plan on making a cream gravy but he's telling me "that's NOT swedish meatballs". I have only been cooking less than a year and he's got a decade experience so I feel like my brain is melting. He can't be right, right?""",
    "weather": weather_str,
    "bjj": f"Here are some rules for Brazilian Jiu-Jitsu. Read them before we roll. And ask if you want me to clarify something.\n{bjj_str}",
    "britney": britney_str,
    "traveler": "I am a lonely traveler. I have travelled far and wide and lost all my money along the way. After months, I finally find myself in this inn talking to a friendly face."
}


def get_opening_question_for_theme(conversation_theme):

    if conversation_theme is None:
        per_participant_context = False
        opening_questions = None

    elif conversation_theme in opening_questions_for_themes:
        per_participant_context = False
        opening_questions = opening_questions_for_themes[conversation_theme]

    elif Path(conversation_theme).is_file():
        per_participant_context = True
        with open(conversation_theme, 'r') as file:
            opening_questions = [json.loads(line)['content'] for line in file]

    else:
        raise ValueError(f"Undefined conversation theme: {conversation_theme}.")

    return opening_questions, per_participant_context


def get_interlocutors(interlocutors):

    if interlocutors == "human":
        per_participant_context = False
        return interlocutors, per_participant_context

    elif Path(interlocutors).is_file():
        per_participant_context = True

        with open(interlocutors, 'r') as f:
            interlocutors = json.load(f)

        return interlocutors, per_participant_context

    else:
        raise ValueError(f"Undefined conversation theme: {interlocutors}.")


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

def simulate_conversation(args, opening_question, model_set_persona_string=None, llm_generator=None, simulated_participant=None, interlocutor="human"):

    conversation = [opening_question]

    # simulate conversation
    assert args.simulated_conversation_n_messages % 2 == 1 # must be odd so that the last one is GPT as simulated persona

    for msg_i in range(args.simulated_conversation_n_messages):
        if args.verbose:
            print(f"\nSimulating message: {msg_i}/{args.simulated_conversation_n_messages}")

        # assign roles to messages - alternating, last one user
        simulated_conv_messages = create_simulated_messages(conversation, last="user")

        interlocutor_name_upper = "HUMAN" if interlocutor == "human" else interlocutor["name"].upper()
        participant_name_upper = simulated_participant["name"].upper()

        assert interlocutor_name_upper != participant_name_upper

        labels_dict = {
            "persona": {
                "assistant_label": participant_name_upper,
                "user_label": "USER",
                "system_label": "CONTEXT"
            },
            "interlocutor": {
                "assistant_label": interlocutor_name_upper,
                "user_label": f"{participant_name_upper} (CHATBOT)" if args.interlocutor_knows_persona else "CHATBOT",
                "system_label": "CONTEXT"
            }
        }
        stop_words_up = [f"\n{v}:" for v in labels_dict["persona"].values()] + [f"\n{v}:" for v in labels_dict["interlocutor"].values()]
        # also add similar words wo whitespace ex. GANDALF (CHATBOT) and GANDALF(CHATBOT)
        stop_words_up += [s.replace(" ", "") for s in stop_words_up if " " in s]

        if msg_i % 2 == 0:
            # even -> model as persona
            assert simulated_conv_messages[0]['role'] == "user"

            if model_set_persona_string:
                simulated_conv_messages = [{
                    "role": "system" if llm_generator.system_message else "user",
                    "content": model_set_persona_string
                }] + simulated_conv_messages

            assistant_label = labels_dict["persona"]["assistant_label"]
            user_label = labels_dict["persona"]["user_label"]
            system_label = labels_dict["persona"]["system_label"]

        else:
            # model as interlocutor
            assert simulated_conv_messages[0]['role'] == "assistant"

            if llm_generator.base_model_template:
                if interlocutor == "human":
                    interlocutor_persona_set_str = "a human"
                else:
                    interlocutor_persona_set_str = interlocutor["description"].removesuffix(".")

                if args.interlocutor_knows_persona:
                    chatbot_persona_str = f"The chatbot is pretending to be {simulated_participant['name']}. "
                else:
                    chatbot_persona_str = ""

                if args.long_messages:
                    n_sent_instr=""
                else:
                    if interlocutor == "human":
                        n_sent_instr = "The human's every reply must be in one sentence only."
                    else:
                        n_sent_instr = f"{interlocutor['name']}'s every reply must be in one sentence only."

                sys_msg = f"The following is a conversation between {interlocutor_persona_set_str} and a chatbot. {chatbot_persona_str}{n_sent_instr}"

            else:

                if interlocutor == "human":
                    interlocutor_persona_set_str = "You are simulating a human using a chatbot. "
                else:
                    interlocutor_persona_set_str = f"You are simulating {interlocutor['description']} "

                if args.interlocutor_knows_persona:
                    chatbot_persona_str = f"The chatbot is pretending to be {simulated_participant['name']}. "
                else:
                    chatbot_persona_str = ""

                if args.long_messages:
                    n_sent_instr = ""
                else:
                    n_sent_instr = "Your every reply must be in one sentence only."

                sys_msg = f"{interlocutor_persona_set_str}{chatbot_persona_str}{n_sent_instr}"

            simulated_conv_messages = [{
                "role": "system" if llm_generator.system_message else "user",
                "content": sys_msg
            }] + simulated_conv_messages

            assistant_label = labels_dict["interlocutor"]["assistant_label"]
            user_label = labels_dict["interlocutor"]["user_label"]
            system_label = labels_dict["interlocutor"]["system_label"]

        if not llm_generator.base_model_template:
            simulated_conv_messages = fix_alternating_msg_order(simulated_conv_messages)

        response = llm_generator.generate(
            messages=simulated_conv_messages,
            assistant_label=assistant_label,
            user_label=user_label,
            system_label=system_label,
            stop_words_up=stop_words_up
        )

        if llm_generator.base_model_template:
            response_up = response.upper()
            stop_word_ind = np.min([response_up.index(sw) if sw in response_up else np.inf for sw in stop_words_up])
            if stop_word_ind != np.inf:
                stop_word_ind = int(stop_word_ind)
                response = response[:stop_word_ind]

        conversation.append(response)

        messages_conv = create_simulated_messages(conversation, last="assistant")

    if args.verbose:
        print("\nSimulated conversation:")
        print_chat_messages(messages_conv)

    return messages_conv

