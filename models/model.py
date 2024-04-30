from abc import ABC, abstractmethod


class Model(ABC):

    def __init__(self, model_id, base_model_template, system_message, verbose=True, **kwargs):
        self.model_id = model_id
        self.base_model_template = base_model_template
        self.system_message = system_message
        self.verbose = verbose

    @abstractmethod
    def predict(self, messages, answer, *args, **kwargs):
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def generate(self, messages, generation_kwargs=None, *args, **kwargs):
        raise NotImplementedError("Not implemented")

