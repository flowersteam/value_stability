import os
from termcolor import colored

from tenacity import retry, stop_after_attempt, wait_random_exponential

from openai import AzureOpenAI
from openai import OpenAI
import tiktoken


from .model import Model
from .utils import *


class OpenAIModel(Model):

    openai_2_azure_tag = {
        "gpt-3.5-turbo-0125": "gpt-35-turbo-0125",
        "gpt-3.5-turbo-1106": "gpt-35-turbo-1106"
    }

    def __init__(self, model_id, use_azure, generation_args, *args, **kwargs):

        super(OpenAIModel, self).__init__(model_id, *args, **kwargs)

        self.azure_id = self.openai_2_azure_tag[self.model_id]
        self.use_azure = use_azure

        if generation_args is None:
            self.generation_args = {}
        else:
            self.generation_args = generation_args

        if self.use_azure:
            print(colored("Using Azure OPENAI API", "red"))

            if self.model_id == "gpt-3.5-turbo-0125":
                self.model = AzureOpenAI(
                    azure_endpoint="https://petunia-grgur.openai.azure.com/",
                    api_key=os.getenv("AZURE_OPENAI_KEY_gpt_35_turbo_0125"),
                    api_version="2024-02-15-preview"
                )

            elif self.model_id == "gpt-3.5-turbo-1106":
                self.model = AzureOpenAI(
                    azure_endpoint="https://petunia-grgur-gpt-35-turbo-1106.openai.azure.com/",
                    api_key=os.getenv("AZURE_OPENAI_KEY_gpt_35_turbo_1106"),
                    api_version="2024-02-15-preview"
                )
            else:
                raise NotImplementedError("Azure endpoint not found.")

        else:
            print(colored("Using OPENAI API", "red"))
            self.model = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    @retry(wait=wait_random_exponential(min=10, max=30), stop=stop_after_attempt(10))
    def completions_with_backoff(self, **kwargs):
        return self.model.chat.completions.create(**kwargs)

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

        messages.append({"role": "assistant", "content": query_string})

        # get the encoding for each letter in choices
        logit_bias = {self.tokenizer.encode(c)[0]: 100 for c in answers}

        c = self.completions_with_backoff(
            model=self.azure_id if self.use_azure else self.model_id,
            messages=messages,
            max_tokens=1,
            n=1,
            temperature=0,
            logit_bias=logit_bias,
        )

        generation = c.choices[0].message.content
        lprobs = dummy_lprobs_from_generation(generation, answers, label_2_text_option_dict)

        return generation, lprobs

    def generate(self, messages, additional_generation_args=None, *args, **kwargs):

        if additional_generation_args is not None:
            generation_args = {**self.generation_args, **additional_generation_args}
        else:
            generation_args = self.generation_args

        c = self.completions_with_backoff(
            model=self.azure_id if self.use_azure else self.model_id,
            messages=messages,
            **generation_args,
        )

        response = c.choices[0].message.content

        if response is None:
            response = " "

        return response

