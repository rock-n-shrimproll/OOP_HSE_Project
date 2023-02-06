from freezing_utils.abstract_freezing import AbstractFreezing


def freeze_all(model):
    """
    Freezes all model layers except for the classifier head.
    For now only works with BERT.
    TODO: avoid hard-coding layer names so it can be used with other encoders.
    """
    for name, param in model.bert.named_parameters():
        param.requires_grad = False
    return model


class FreezeAll(AbstractFreezing):
    def freeze(self, model):
        return freeze_all(model)
