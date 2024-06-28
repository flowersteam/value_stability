import os
import torch
import json
import importlib

from .utils import *
from .model import Model
from .dummymodel import DummyModel
from .interactivemodel import InteractiveModel
from .openaimodel import OpenAIModel
from .huggingfacemodel import *

from transformers import BitsAndBytesConfig

hf_token = os.environ["HF_TOKEN"]


def load_model_args(model_config_path):

    try:
        with open(model_config_path, 'r') as file:
            model_args = json.load(file)

    except FileNotFoundError:
        raise FileNotFoundError(f"The configuration file {model_config_path} could not be found.")

    if 'load_args' in model_args:
        # parse hf token
        if "token" in model_args['load_args']:
            if model_args['load_args']['token'] == "HF_TOKEN":
                model_args['load_args']['token'] = hf_token

        # parse torch.dtype
        if "torch_dtype" in model_args['load_args']:
            if model_args['load_args']['torch_dtype'].startswith("torch."):
                model_args['load_args']['torch_dtype'] = eval(model_args['load_args']['torch_dtype'])

        if "quantization_config" in model_args['load_args']:
            model_args['load_args']['quantization_config'] = eval(model_args['load_args']['quantization_config'])

    # load model class
    my_module = importlib.import_module("models")
    ModelClass = getattr(my_module, model_args['model_class'])

    return ModelClass, model_args


def create_model(model_config_path):
    ModelClass, model_args = load_model_args(model_config_path)
    return ModelClass(**model_args)