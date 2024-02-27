from abc import ABC, abstractmethod
from model import GLM
import sys


class QAModelFactory(ABC):

    @abstractmethod
    def create_model(self, model_params):
        raise NotImplementedError()

    @abstractmethod
    def add_to_path(self):
        raise NotImplementedError()

    def create(self, model_params):
        self.add_to_path()
        return self.create_model(model_params)


class GLMFactory(QAModelFactory):

    def add_to_path(self):
        sys.path.append("./glm")

    def create_model(self, model_params):
        return GLM(**model_params)

