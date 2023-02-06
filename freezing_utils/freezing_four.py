from freezing_utils.abstract_freezing import AbstractFreezing
from base_logic.utils import freeze_some_layers


class FreezeFour(AbstractFreezing):
    def freeze(self, model):
        return freeze_some_layers(model, 4)
