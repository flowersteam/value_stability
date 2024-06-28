import random
import tiktoken

from .model import Model
from .utils import *

class DummyModel(Model):
    def __init__(self, model_id, *args, **kwargs):

        super(DummyModel, self).__init__(
            model_id=model_id,
            *args, **kwargs
        )
        self.gpt_tokenizer = tiktoken.get_encoding("cl100k_base")

    def generate(self, messages, assistant_label=None, *args, **kwargs):

        messages = messages[:]

        response = f"Dummy simulated message. This is a filler message it same some extra text so as to help estimate the number of tokens. As the gpt generations is set to 100 tokens max. Here we aim to also 100 tokens message. I am repeating it now. This is a filler message it same some extra text so as to help estimate the number of tokens. As the gpt generations is set to 100 tokens max. Here we aim to also 100 tokens message."

        if self.verbose:
            # formatted_prompt = apply_base_model_template(
            #     messages,
            #     add_generation_prompt=True,
            #     assistant_label=assistant_label,
            #     user_label="USER",
            #     system_label="CONTEXT"
            # )
            n_tokens = sum([len(self.gpt_tokenizer.encode(m['content'])) for m in messages])
            print(f">>> {self.__class__}.generate (input tokens: {n_tokens})")
            print_chat_messages(messages)
            # print(f"************************FORMATTED PROMPT********************\n{formatted_prompt}\n******************")
            print(f"-(generation)->{response}")

        return response

    def predict(
        self,
        messages,
        answers,
        label_2_text_option_dict=None,
        query_string=None,
        assistant_label=None,
        user_label="USER",
        system_label="CONTEXT",
        *args, **kwargs
    ):
        messages = messages[:]
        if label_2_text_option_dict is None:
            raise ValueError("label_2_text_option_dict must be provided")

        if assistant_label is None:
            raise ValueError("assistant_label must be provided. ")

        # formatted_prompt = apply_base_model_template(
        #     messages,
        #     add_generation_prompt=True,
        #     assistant_label=assistant_label,
        #     user_label="USER",
        #     system_label="CONTEXT"
        # )

        messages += [{
            "role": "assistant",
            "content": query_string
        }]

        if self.verbose:
            n_tokens = sum([len(self.gpt_tokenizer.encode(m['content'])) for m in messages])
            print(f">>> {self.__class__}.predict (input tokens: {n_tokens})")
            print_chat_messages(messages)

            # print(f"************************FORMATTED PROMPT*********************\n{formatted_prompt}\n******************")

        import re
        try:
            msg_lines = messages[-2]['content'].split("\n")
            if messages[0]['content'].startswith("You are M") and \
                any([l.startswith("AMBITIOUS") for l in msg_lines]) or \
                any([l.startswith("INFLUENTIAL") for l in msg_lines]) or \
                any([l.startswith("CAPABLE") for l in msg_lines]) or \
                any([l.startswith("SUCCESSFUL") for l in msg_lines]):
                generation = messages[-2]['content'][messages[-2]['content'].index(") 7") - 1:][:1]

            else:
                # generation = messages[-2]['content'][messages[-2]['content'].index(") -1") - 1:][:1]
                generation = random.choice([f"{c}" for c in answers])

        except:
            generation = random.choice([f"{c}" for c in answers])

        # generation = random.choice([f"{c}" for c in answers])

        if self.verbose:
            print(f"-(generation)->{generation}")

        lprobs = dummy_lprobs_from_generation(generation, answers, label_2_text_option_dict)

        return generation, lprobs
