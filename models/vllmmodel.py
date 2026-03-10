import time
import warnings
import os

from openai import OpenAI


from .model import Model
from .utils import *



class APIModel(Model):

    def __init__(self, model_id, generation_args, load_args, cot, *args, **kwargs):

        super(APIModel, self).__init__(model_id, *args, **kwargs)

        if generation_args is None:
            self.generation_args = {}
        else:
            self.generation_args = generation_args

        self.cot = cot
        if self.cot:
            self.query_string = '\nThink step by step and then finish your answer with "Answer: (X)", where X is the letter of the option.'
        else:
            self.query_string = '\nReply ONLY with "(X)", where X is the letter of the option.'

        self.load_args = load_args


    # @retry(wait=wait_random_exponential(min=10, max=30), stop=stop_after_attempt(10))
    def completions_with_backoff(self, **kwargs):
        return self.model.chat.completions.create(**kwargs)

    def extract_generation_reasoning_and_response(self, c):
        # CoT : "Answer:"
        generation = c.choices[0].message.content

        if "Answer:" not in generation:
            warnings.warn(f'generation does not contain "Answer:". Generation: {generation }')
            reasoning, response = "error", "error"  # random option will be selected
        elif generation .count("Answer:") > 1:
            warnings.warn(f'Multiple "Answer:" tags present in generation. Generation: {generation }')
            reasoning, response = "error", "error"  # random option will be selected
        else:
            reasoning, response = generation.split("Answer:")

        return reasoning, response

    def predict(
        self,
        messages,
        answers,
        # query_string=None,
        label_2_text_option_dict=None,
        return_full_generation=False,
        *args, **kwargs
    ):
        if label_2_text_option_dict is None:
            raise ValueError("label_2_text_option_dict must be provided")

        messages = messages[:]
        messages[-1]['content'] += self.query_string

        if self.verbose:
            print(f">>> {self.__class__}.predict")
            print_chat_messages(messages)

        max_response_tokens = 10

        if self.cot:
            c = self.completions_with_backoff(
                model=self.model_id,
                messages=messages,
                temperature=0,
                max_tokens=2000,
            )

            # both CoT and response,we return this so that svs has examples of CoT
            generation = c.choices[0].message.content
            reasoning, response = self.extract_generation_reasoning_and_response(c)

            # crop generation to be fair to non-reasoning models
            max_characters = max_response_tokens * 4  # rule of thumb: 4 characters per token
            response = response[:max_characters]

            if self.verbose:
                print(colored(f"-(reasoning-L:{len(reasoning)})->{reasoning}", "green"))
                print(colored(f"-(response)->{response}", "green"))

            # parse only response
            lprobs, match = dummy_lprobs_from_generation(response, answers, label_2_text_option_dict)

        else:
            c = self.completions_with_backoff(
                model=self.model_id,
                messages=messages,
                temperature=0,
                max_tokens=max_response_tokens,
            )
            generation = c.choices[0].message.content

            if self.verbose:
                print(colored(f"-(generation)->{generation}", "green"))

            lprobs, match = dummy_lprobs_from_generation(generation, answers, label_2_text_option_dict)

        return generation, lprobs, match

    def generate(self, messages, additional_generation_args=None, *args, **kwargs):

        messages = messages[:]

        if additional_generation_args is not None:
            generation_args = {**self.generation_args, **additional_generation_args}
        else:
            generation_args = self.generation_args.copy()

        if self.verbose:
            print(f">>> {self.__class__}.generate")
            print_chat_messages(messages)

        c = self.completions_with_backoff(
            model=self.model_id,
            messages=messages,
            **generation_args,
        )

        generation = c.choices[0].message.content

        if generation is None:
            generation = " "

        if self.verbose:
            print(colored(f"-(generation)->{generation}", "green"))

        return generation


class VllmModel(APIModel):

    def __init__(self, model_id, generation_args, load_args, cot, *args, **kwargs):

        super(VllmModel, self).__init__(model_id, generation_args, load_args, cot, *args, **kwargs)

        print(colored("Using vLLM model", "red"))
        time.sleep(1)

        self.load_args["api_key"] = "EMPTY"
        self.load_args["base_url"] = "http://localhost:8000/v1"

        self.model = OpenAI(**self.load_args)
class DeepSeekModel(APIModel):

    def __init__(self, model_id, generation_args, load_args, cot, *args, **kwargs):

        super(DeepSeekModel, self).__init__(model_id, generation_args, load_args, cot, *args, **kwargs)

        print(colored("Using DeepSeek model", "red"))
        time.sleep(1)

        self.load_args["api_key"] = os.getenv("DS_API_KEY")
        self.load_args["base_url"] = "https://api.deepseek.com"
        self.model = OpenAI(**self.load_args)
