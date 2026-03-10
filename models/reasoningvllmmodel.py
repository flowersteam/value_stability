import abc
import time
import warnings

from tenacity import retry, stop_after_attempt, wait_random_exponential

from openai import OpenAI


from .model import Model
from .utils import *


class ReasoningVllmModel(Model):

    def __init__(self, model_id, generation_args, load_args, *args, **kwargs):

        super(ReasoningVllmModel, self).__init__(model_id, *args, **kwargs)

        if generation_args is None:
            self.generation_args = {}
        else:
            self.generation_args = generation_args

        print(colored("Using vLLM", "red"))
        time.sleep(1)

        # default load args
        self.load_args = {
            **{"api_key": "EMPTY", "base_url": "http://localhost:8000/v1"},
            **load_args
        }
        self.model = OpenAI(**self.load_args)

        self.query_string = '\nReply ONLY with "(X)", where X is the letter of the option.'

    @abc.abstractmethod
    def extract_reasoning_and_generation(self, c):
        "This usually doesn't work for open-weight models, so it should be implemented in the child class"
        reasoning = c.choices[0].message.reasoning_content
        generation = c.choices[0].message.content
        return reasoning, generation

    # @retry(wait=wait_random_exponential(min=10, max=30), stop=stop_after_attempt(10))
    def completions_with_backoff(self, **kwargs):
        return self.model.chat.completions.create(**kwargs)

    def predict(
            self,
            messages,
            answers,
            # query_string=None,
            label_2_text_option_dict=None,
            *args, **kwargs
    ):
        if label_2_text_option_dict is None:
            raise ValueError("label_2_text_option_dict must be provided")

        messages = messages[:]
        messages[-1]['content'] += self.query_string

        if self.verbose:
            print(f">>> {self.__class__}.predict")
            print_chat_messages(messages)

        c = self.completions_with_backoff(
            model=self.model_id,
            messages=messages,
            temperature=0,
            max_tokens=2000,
        )

        reasoning, generation = self.extract_reasoning_and_generation(c)

        if generation is None:
            generation = " "

        # crop generation to be fair to non-reasoning models
        max_tokens = 10
        max_characters = max_tokens * 4  # rule of thumb: 4 characters per token
        generation = generation[:max_characters]

        if self.verbose:
            print(f"-(reasoning)->{reasoning}")
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
            generation_args = self.generation_args.copy()

        max_generation_tokens = generation_args["max_generation_tokens"]
        del generation_args["max_generation_tokens"]

        c = self.completions_with_backoff(
            model=self.model_id,
            messages=messages,
            max_tokens=2000,  # upper limit for reasoning and generation
            **generation_args,
        )

        reasoning, response = self.extract_reasoning_and_generation(c)

        if response is None:
            response = " "

        # crop generation to be fair to non-reasoning models
        max_characters = max_generation_tokens * 4  # rule of thumb: 4 characters per token
        response = response[:max_characters]

        if self.verbose:
            print(f"-(reasoning)->{reasoning}")
            print(f"-(generation)->{response}")

        return response


class Qwen3ReasoningVllmModel(ReasoningVllmModel):

    def extract_reasoning_and_generation(self, c):

        # Qwen3 : </think>
        reasoning = c.choices[0].message.reasoning_content
        generation = c.choices[0].message.content

        if reasoning is None:
            # it is possible that reasoning was not correctly extracted (bug with QwQ)
            if "</think>" not in generation:
                warnings.warn(f"reasoning is None and generation does not contain </think>. Generation: {generation}")
                reasoning, generation = "error", "error"  # random option will be selected
            elif generation.count("</think>") > 1:
                warnings.warn(f"Multiple </think> tags present in generation. Generation: {generation}")
                reasoning, generation = "error", "error"  # random option will be selected
            else:
                reasoning, generation = generation.split("</think>")

        # if generation is None:
        #     from IPython import embed; embed()

        return reasoning, generation


class QwQReasoningVllmModel(ReasoningVllmModel):

    def extract_reasoning_and_generation(self, c):
        # QWQ : </think>
        reasoning = c.choices[0].message.reasoning_content
        generation = c.choices[0].message.content

        if reasoning is None:
            # it is possible that reasoning was not correctly extracted (bug with QwQ)
            if "</think>" not in generation:
                warnings.warn(f"reasoning is None and generation does not contain </think>. Generation: {generation}")
                reasoning, generation = "error", "error"  # random option will be selected
            elif generation.count("</think>") > 1:
                warnings.warn(f"Multiple </think> tags present in generation. Generation: {generation}")
                reasoning, generation = "error", "error"  # random option will be selected
            else:
                reasoning, generation = generation.split("</think>")


        return reasoning, generation

class RekaReasoningVllmModel(ReasoningVllmModel):

    def extract_reasoning_and_generation(self, c):
        # Reka : </reasoning> and <sep>
        reasoning = c.choices[0].message.reasoning_content
        generation = c.choices[0].message.content

        if reasoning is None:
            # it is possible that reasoning was not correctly extracted (bug with QwQ)
            if "</reasoning>" not in generation:
                warnings.warn(f"reasoning is None and generation does not contain </reasoning>. Generation: {generation}")
                reasoning, generation = "error", "error"  # random option will be selected
            elif generation.count("</reasoning>") > 1:
                warnings.warn(f"Multiple </reasoning> tags present in generation. Generation: {generation}")
                reasoning, generation = "error", "error"  # random option will be selected
            else:
                reasoning, generation = generation.split("</reasoning>")

            generation = generation.split("<sep>")[0]

        return reasoning, generation
