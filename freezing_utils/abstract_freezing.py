from abc import ABC, abstractmethod


class AbstractFreezing(ABC):

    @abstractmethod
    def freeze(self, model):
        pass
