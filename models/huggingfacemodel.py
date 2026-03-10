import os
import warnings

from .model import Model
from .utils import *
import time

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
import numpy as np

try:
    from vllm import LLM, SamplingParams
except:
    pass


# hf_cache_dir = get_hf_cache_dir()
# os.environ['HF_HOME'] = hf_cache_dir


class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops, tokenizer, original_input_ids):
        super().__init__()
        self.stops = [s.upper() for s in stops]
        self.tokenizer = tokenizer
        self.original_input_ids = original_input_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs):
        generated_ids = input_ids[0][len(self.original_input_ids[0]):]
        generation = self.tokenizer.decode(generated_ids).upper()
        return any([stop in generation for stop in self.stops])


class HuggingFaceModel(Model):

    def __init__(
            self,
            model_id,
            base_model_template,
            system_message,
            load_args=None,
            generation_args=None,
            tokenizer_load_args=None,
            use_vllm=False,
            *args,
            **kwargs
    ):
        raise DeprecationWarning("Update according to openroutermodel.py -> free generation")
        super(HuggingFaceModel, self).__init__(
            model_id=model_id,
            base_model_template=base_model_template,
            system_message=system_message,
            *args, **kwargs
        )

        self.use_vllm = use_vllm

        if load_args is None:
            self.load_args = {}
        else:
            self.load_args = load_args
        print("Model Load args:", self.load_args)

        if tokenizer_load_args is None:
            self.tokenizer_load_args = self.load_args
        else:
            self.tokenizer_load_args = tokenizer_load_args

        print("Tokenizer Load args:", self.tokenizer_load_args)

        if generation_args is None:
            self.generation_args = {}
        else:
            self.generation_args = generation_args
        print("Generation args:", self.generation_args)

        start_time = time.time()

        if self.use_vllm:
            self.model = LLM(model=self.model_id, **self.load_args, seed=np.random.randint(0, 1e9))
            self.tokenizer = self.model.get_tokenizer()

        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                **self.tokenizer_load_args,
                # cache_dir=hf_cache_dir,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                **self.load_args,
                # cache_dir=hf_cache_dir
            ).eval()
        print(f"Model loaded: {self.model_id}")
        end_time = time.time()

        # set the max context length
        if hasattr(self.model.config, "model_max_length"):
            self.max_context_length = self.model.config.model_max_length
        elif hasattr(self.model.config, "max_position_embeddings"):
            self.max_context_length = self.model.config.max_position_embeddings
        else:
            self.max_context_length = self.model.config.text_config.max_position_embeddings

        if self.verbose:
            print("Model loading time: {}h {}m {}s".format(*secs_2_hms(end_time-start_time)))

    def extract_answer_tokens(self, answers):
        answer_tokens = {a: [] for a in answers}
        for tok_ind in range(len(self.tokenizer)):
            tok = self.tokenizer.decode([tok_ind])
            if tok in answers:
                answer_tokens[tok].append(tok_ind)

        return answer_tokens

    def parse_hf_outputs(self, output, answers):

        answer_tokens = self.extract_answer_tokens(answers)  # todo: repetitive -> extract

        option_scores = {
            ans: max([output.scores[0][0, ind] for ind in answer_tokens[ans]])
            for ans in answers
        }

        # take the most probable answer as the generation
        generation = max(option_scores, key=option_scores.get)

        # extract logprobs
        lprobs = [float(option_scores[a]) for a in answers]

        # todo: check that ' A' are one token and check for those as well and not "unk"
        encoded_ans = [self.tokenizer.encode(ans, add_special_tokens=False)[0] for ans in answers]
        option_scores = {enc_a: output.scores[0][0, enc_a] for enc_a in encoded_ans}

        return option_scores, generation, lprobs

    def predict(
        self,
        messages,
        answers,
        query_string=None,
        assistant_label=None,
        user_label="USER",
        system_label="CONTEXT",
        *args, **kwargs
    ):
        messages = messages[:]
        messages[-1]['content'] += query_string

        if self.base_model_template:
            if assistant_label is None:
                raise ValueError("assistant_label must be provided with base model template.")

            formatted_prompt = apply_base_model_template(
                messages,
                add_generation_prompt=True,
                assistant_label=assistant_label,
                user_label="USER",
                system_label="CONTEXT"
            )

        else:
            formatted_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # formatted_prompt += query_string # query string in assistant

        if self.verbose:
            print(f">>> {self.__class__}.predict")
            if self.base_model_template:
                print(f"************************FORMATTED PROMPT*********************\n{formatted_prompt}\n******************")
            else:
                print_chat_messages(messages+[{"role": "assistant", "content": query_string}])

        if self.use_vllm:
            outputs = self.model.generate([formatted_prompt], SamplingParams(max_tokens=1, logprobs=5))

            tok_lprobs = outputs[0].outputs[0].logprobs[0]
            tok_2_lprobs = {tok_lprob.decoded_token: tok_lprob.logprob for tok_lprob in tok_lprobs.values()}

            generation = max(tok_2_lprobs, key=tok_2_lprobs.get)
            lprobs = [tok_2_lprobs.get(a, -100) for a in answers]

            if lprobs == [-100]*len(answers):
                raise ValueError("No answer was given. vLLM allows only top 5 logprobs, you need to use the transformers library (use_vllm=False) with this model")

        else:
            inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.model.device)


            if inputs['input_ids'].numel() > self.max_context_length:
                warnings.warn(f"Input ({inputs['input_ids'].numel()}) exceeds max context length of {self.max_context_length}.")

            # token match
            output = self.model.generate(
                **inputs,
                max_new_tokens=1,
                return_dict_in_generate=True,
                output_scores=True
            )

            _, generation, lprobs = self.parse_hf_outputs(output=output, answers=answers)

        if self.verbose:
            print(f"-(generation)->{generation}")

        return generation, lprobs

    def generate(
            self,
            messages,
            additional_generation_args=None,
            assistant_label=False,
            user_label=False,
            system_label=False,
            stop_words_up=None
    ):
        messages = messages[:]

        if self.verbose:
            print(f">>> {self.__class__}.generate")

        if self.base_model_template:
            if self.use_vllm:
                raise NotImplementedError("VLLM not implemented for base models")

            if not self.system_message:
                raise ValueError("system_message must be used with base model template")

            if assistant_label is None:
                raise ValueError("assistant_label must be defined with base model template")

            if user_label is None:
                raise ValueError("user_label must be defined with base model template")

            if system_label is None:
                raise ValueError("system_label must be defined with base model template")

            if stop_words_up is None:
                raise ValueError("stop_words_up must be defined with base model template (Uppercase stop words)")

            formatted_prompt, stop_words = apply_base_model_template(
                messages,
                assistant_label=assistant_label,
                user_label=user_label,
                system_label=system_label,
                add_generation_prompt=True,
                return_stop_words=True
            )
            input_ids = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.model.device).input_ids

            if input_ids.numel() > self.max_context_length:
                warnings.warn(f"Input ({input_ids.numel()}) exceeds max context length of {self.max_context_length}.")

            assert all([w.upper() in stop_words_up for w in stop_words])
            stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stop_words_up, self.tokenizer, input_ids)])

            if self.verbose:
                print(f"************************FORMATTED PROMPT********************\n{formatted_prompt}\n******************")

        else:
            if self.use_vllm:
                formatted_messages = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )

            else:
                input_ids = self.tokenizer.apply_chat_template(
                    messages,
                    return_tensors="pt",
                    add_generation_prompt=True
                ).to(self.model.device)

            if self.verbose:
                print_chat_messages(messages)

            stopping_criteria = None

        if additional_generation_args is not None:
            generation_args = {**self.generation_args, **additional_generation_args}
        else:
            generation_args = self.generation_args

        if self.use_vllm:
            outputs = self.model.generate(
                [formatted_messages],
                SamplingParams(**generation_args)
            )
            response = outputs[0].outputs[0].text

        else:
            if input_ids.numel() > self.max_context_length:
                warnings.warn(f"Input ({input_ids.numel()}) exceeds max context length of {self.max_context_length}.")

            output_seq = self.model.generate(
                input_ids=input_ids,
                **generation_args,
                return_dict_in_generate=True,
                output_scores=True,
                stopping_criteria=stopping_criteria
            )

            response = self.tokenizer.decode(output_seq.sequences[0][len(input_ids[0]):], skip_special_tokens=True)

        if self.verbose:
            print(f"-(generation)->{response}")

        return response


class LLama3Model(HuggingFaceModel):

    def __init__(self, *args, **kwargs):
        super(LLama3Model, self).__init__(*args, **kwargs)

        self.generation_args["eos_token_id"] = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]


try:
    from mistral_common.tokens.instruct.normalize import ChatCompletionRequest
    from mistral_common.protocol.instruct.messages import AssistantMessage, UserMessage
except:
    ...

def to_mistral_msg(msg):

    if msg['role'] == 'user':
        return UserMessage(content=msg['content'])
    elif msg['role'] == 'assistant':
        return AssistantMessage(content=msg['content'])
    else:
        raise ValueError(f"Undefined message role {msg['role']}")


class MistralSmallInstruct2409Model(HuggingFaceModel):

    def __init__(self, *args, **kwargs):
        super(MistralSmallInstruct2409Model, self).__init__(*args, **kwargs)
        self.tokenizer.pad_token = self.tokenizer.eos_token


class Mixtral8x22BModel(HuggingFaceModel):

    def generate(self, messages, *args, **kwargs):

        if not self.base_model_template:
            mistral_query = ChatCompletionRequest(messages=list(map(to_mistral_msg, messages)), model="test")
            messages = mistral_query.model_dump()['messages']

        return super(Mixtral8x22BModel, self).generate(messages, *args, **kwargs)
