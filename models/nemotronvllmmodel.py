import time

from tenacity import retry, stop_after_attempt, wait_random_exponential

from openai import OpenAI


from .model import Model
from .utils import *


class NemotronVllmModel(Model):

    def __init__(self, model_id, generation_args, load_args, reasoning, *args, **kwargs):

        super(NemotronVllmModel, self).__init__(model_id, *args, **kwargs)
        raise DeprecationWarning()

        if generation_args is None:
            self.generation_args = {}
        else:
            self.generation_args = generation_args

        print(colored("Using vLLM", "red"))
        time.sleep(1)

        self.reasoning = reasoning
        self.thinking = "on" if self.reasoning else "off"

        # default load args
        self.load_args = {
            **{"api_key": "EMPTY", "base_url": "http://localhost:8000/v1"},
            **load_args
        }
        self.model = OpenAI(**self.load_args)

    # @retry(wait=wait_random_exponential(min=10, max=30), stop=stop_after_attempt(10))
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

        messages = messages[:]

        messages[0]['content'] += f"\ndetailed thinking {self.thinking}"
        assert messages[0]['role'] == "system"

        messages[-1]['content'] += query_string

        if self.verbose:
            print(f">>> {self.__class__}.predict")
            print_chat_messages(messages)

        c = self.completions_with_backoff(
            model=self.model_id,
            messages=messages,
            # max_tokens=1,
            # max_tokens=200,
            max_tokens=10,
            temperature=0,
        )

        generation = c.choices[0].message.content

        if self.verbose:
            print(f"-(generation)->{generation}")

        lprobs, match = dummy_lprobs_from_generation(generation, answers, label_2_text_option_dict)

        return generation, lprobs, match

    def generate(self, messages, additional_generation_args=None, *args, **kwargs):

        messages = messages[:]
        messages[0]['content'] += f"\ndetailed thinking {self.thinking}"
        assert messages[0]['role'] == "system"


        if self.verbose:
            print(f">>> {self.__class__}.generate")
            print_chat_messages(messages)

        if additional_generation_args is not None:
            generation_args = {**self.generation_args, **additional_generation_args}
        else:
            generation_args = self.generation_args

        c = self.completions_with_backoff(
            model=self.model_id,
            messages=messages,
            **generation_args,
        )

        response = c.choices[0].message.content

        if response is None:
            response = " "

        if self.verbose:
            print(f"-(generation)->{response}")

        return response

