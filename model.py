from abc import ABC, abstractmethod


class QAModel(ABC):

    @abstractmethod
    def train(self):
        raise NotImplementedError()

    @abstractmethod
    def infer_val(self):
        raise NotImplementedError()

    @abstractmethod
    def infer_test(self):
        raise NotImplementedError()


class GLM(QAModel):
    pass