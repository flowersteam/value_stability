import random

from .model import Model
from .utils import *

class DummyModel(Model):
    def __init__(self, model_id, *args, **kwargs):

        super(DummyModel, self).__init__(
            model_id=model_id,
            *args, **kwargs
        )

    def generate(self, *args, **kwargs):
        response = f"Dummy simulated message. This is a filler message it same some extra text so as to help estimate the number of tokens. As the gpt generations is set to 100 tokens max. Here we aim to also 100 tokens message. I am repeating it now. This is a filler message it same some extra text so as to help estimate the number of tokens. As the gpt generations is set to 100 tokens max. Here we aim to also 100 tokens message."
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
        if label_2_text_option_dict is None:
            raise ValueError("label_2_text_option_dict must be provided")

        if assistant_label is None:
            raise ValueError("assistant_label must be provided. ")

        formatted_prompt = apply_base_model_template(
            messages,
            add_generation_prompt=True,
            assistant_label=assistant_label,
            user_label="USER",
            system_label="CONTEXT"
        )

        messages += [{
            "role": "assistant",
            "content": query_string
        }]

        if self.verbose:
            print(f"************************\nFORMATTED PROMPT:\n{formatted_prompt}\n******************")

        # import re
        # generation = messages[-2]['content'][messages[-2]['content'].index(") a few hours per day") - 1:][:1]
        generation = random.choice([f"{c}" for c in answers])

        lprobs = dummy_lprobs_from_generation(generation, answers, label_2_text_option_dict)

        return generation, lprobs
