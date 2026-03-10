import os
import time

from tenacity import retry, stop_after_attempt, wait_random_exponential
from transformers import AutoTokenizer

import requests
import json


from .model import Model
from .utils import *


class OpenRouterModel(Model):

    def __init__(self, model_id, generation_args, api_args, *args, **kwargs):

        super(OpenRouterModel, self).__init__(model_id, *args, **kwargs)

        if generation_args is None:
            self.generation_args = {}
        else:
            self.generation_args = generation_args

        if api_args is None:
            self.api_args = {}
        else:
            self.api_args = api_args

        self.api_key = os.environ.get('OPENROUTER_API_KEY')

        print(colored("Using OpenRouter API", "red"))
        print(f"Model: {self.model_id}")

    def predict(
        self,
        messages,
        answers,
        query_string=None,
        label_2_text_option_dict=None,
        *args, **kwargs
    ):
        if label_2_text_option_dict is None:
            raise ValueError("label_2_text_option_dict must be provided")

        messages = messages[:]

        messages[-1]['content'] += query_string

        if self.verbose:
            print(f">>> {self.__class__}.predict")
            print_chat_messages(messages)

        c = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            data=json.dumps({
                "model": self.model_id,
                "messages": messages,
                "n": 1,
                "temperature": 0,
                "max_tokens": 1,
                # "max_tokens": 300, # freeform
                **self.api_args,
            })
        ).json()
        generation = c['choices'][0]['message']['content']

        if self.verbose:
            print(f"-(generation)->{generation}")

        lprobs, match = dummy_lprobs_from_generation(generation, answers, label_2_text_option_dict)

        return generation, lprobs, match

    def generate(self, messages, additional_generation_args=None, *args, **kwargs):

        messages = messages[:]

        if self.verbose:
            print(f">>> {self.__class__}.generate")
            print_chat_messages(messages)

        if additional_generation_args is not None:
            generation_args = {**self.generation_args, **additional_generation_args}
        else:
            generation_args = self.generation_args

        c = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            data=json.dumps({
                "model": self.model_id,
                "messages": messages,
                **generation_args,
                **self.api_args,
            })
        ).json()
        response = c['choices'][0]['message']['content']

        if response is None:
            response = " "

        if self.verbose:
            print(f"-(generation)->{response}")

        return response

