from freezing_utils.abstract_freezing import AbstractFreezing


class FreezeNone(AbstractFreezing):
    def freeze(self, model):
        return model
