from freezing_utils.freezing_none import FreezeNone
from freezing_utils.freezing_two import FreezeTwo
from freezing_utils.freezing_four import FreezeFour
from freezing_utils.freezing_all import FreezeAll
from freezing_utils.abstract_freezing import AbstractFreezing

ALLOW_FULL_LIST = {'bert-base-uncased', 'bert-base-cased-conv'}


class FreezingFactory:

    @staticmethod
    def get_freezing_object(conf_freeze, encoder) -> AbstractFreezing:
        # print(conf_freeze, encoder)
        if conf_freeze == 'full':
            if encoder not in {'bert-base-uncased', 'bert-base-cased-conv'}:
                raise ValueError("This model does not allow full freezing")
                # print("This model does not allow full freezing")
            else:
                return FreezeAll()
        elif conf_freeze == '4_frozen':
            return FreezeFour()
        elif conf_freeze == '2_frozen':
            return FreezeTwo()
        elif conf_freeze == 'none':
            return FreezeNone()
        else:
            raise ValueError("Unexpected value in factory")
            # print("Unexpected value in factory")
