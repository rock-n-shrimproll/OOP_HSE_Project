from unittest import TestCase
import freezing_factory
import freezing_utils
from freezing_utils.freezing_none import FreezeNone
from freezing_utils.freezing_two import FreezeTwo
from freezing_utils.freezing_all import FreezeAll


class TestFreezingFactory(TestCase):
    def test_get_freezing_object_incorrect_encoder(self):
        print("Тест-1. Некорректная модель")
        with self.assertRaises(ValueError) as context:
            freezing_factory.FreezingFactory.get_freezing_object('full', 'кря')
        self.assertEqual("This model does not allow full freezing", str(context.exception))

    def test_get_freezing_object_incorrect_freezing(self):
        print("Тест-2. Некорректная заморозка")
        with self.assertRaises(ValueError) as context:
            freezing_factory.FreezingFactory.get_freezing_object('кря', 'bert-base-uncased')
        self.assertEqual("Unexpected value in factory", str(context.exception))

    def test_get_freezing_object_non_bert_full_freezing(self):
        print("Тест-3. Разморозка всех слоев для НЕ bert")
        with self.assertRaises(ValueError) as context:
            freezing_factory.FreezingFactory.get_freezing_object('full', 'roberta-base')
        self.assertEqual("This model does not allow full freezing", str(context.exception))

    def test_get_freezing_object_correct_func_from_factory(self):
        print("Тест-4. Фабрика вернула правильную функцию")
        self.assertEqual(type(freezing_factory.FreezingFactory.get_freezing_object('2_frozen', 'bert-base-uncased')),
                         type(freezing_utils.freezing_two.FreezeTwo()))

    def test_get_freezing_object_correct_func_from_factory_bert_full_freeze(self):
        print("Тест-5. Фабрика вернула правильную функцию для full заморозки модели bert")
        self.assertEqual(type(freezing_factory.FreezingFactory.get_freezing_object('full', 'bert-base-cased-conv')),
                         type(freezing_utils.freezing_all.FreezeAll()))

    def test_get_freezing_object_correct_func_from_factory_none_as_default(self):
        print("Тест-6. Фабрика вернула правильную функцию для дефолтной заморозки")
        self.assertEqual(type(freezing_factory.FreezingFactory.get_freezing_object('none', 'roberta-base-conv')),
                         type(freezing_utils.freezing_none.FreezeNone()))
