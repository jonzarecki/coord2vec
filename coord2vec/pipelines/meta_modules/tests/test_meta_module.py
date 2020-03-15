import os
import pickle
from unittest import TestCase

from coord2vec.pipelines.meta_modules.meta_module import MetaModule


class MockMetaModule_doesnt_define_param_in_init(MetaModule):
    def __init__(self, **kwargs):
        super(MockMetaModule_doesnt_define_param_in_init, self).__init__(**kwargs)

    hp_param1 = {
        'a': 1,
        'b': 2
    }


class MockMetaModule_with_object_in_params(MetaModule):
    def __init__(self, param1, **kwargs):
        super(MockMetaModule_with_object_in_params, self).__init__(**kwargs)

    hp_param1 = {
        'a': Warning(),
    }


class MockMetaModule_with_heirarchy(MetaModule):
    def __init__(self, param1, param1_a__param2=2, param1_a__param2_1__param3=3, **kwargs):
        super(MockMetaModule_with_heirarchy, self).__init__(**kwargs)

    hp_param1 = {
        'a': lambda: 2,
        'b': lambda: 1
    }

    hp_param1_a__param2 = [1, 2]
    hp_param1_a__param2_1__param3 = [3,4]


class MockMetaModule_with_implicit_condition(MetaModule):
    def __init__(self, param1, param1_a__param2=2, param2_1__param3=3, **kwargs):
        super(MockMetaModule_with_implicit_condition, self).__init__(**kwargs)

    hp_param1 = {
        'a': lambda: 2,
        'b': lambda: 1
    }

    hp_param1_a__param2 = [1, 2]
    hp_param2_1__param3 = [3,4]

class TestMetaModule(TestCase):
    def test_MockMetaModule_doesnt_define_param_in_init_fails(self):
        with self.assertRaises(AssertionError):
            a = MockMetaModule_doesnt_define_param_in_init()

    # TODO: add test to check if additional param in __init__ fails


    def test_MockMetaModule_fails_with_object_in_hp(self):
        with self.assertRaises(AssertionError):
            a = MockMetaModule_with_object_in_params(1)

    def test_MockMetaModule_returns_correct_hyperparams(self):
        combs = MockMetaModule_with_heirarchy.get_all_hyperparameter_combinations()
        to_tpl = lambda d: tuple(d.items())
        self.assertSetEqual({to_tpl(dict(param1='a', param1_a__param2=1, param1_a__param2_1__param3=3)),
                             to_tpl(dict(param1='a', param1_a__param2=1, param1_a__param2_1__param3=4)),
                             to_tpl(dict(param1='a', param1_a__param2=2)),
                             to_tpl(dict(param1='b'))}, set([to_tpl(d) for d in combs]))

    # TODO: don't have implementation to check that values are in accepted choices

    # TODO: add test to check object with lambda can be pickled
    def test_if_module_with_lambda_can_be_pickled(self):
        a = MockMetaModule_with_heirarchy(1)
        with open("t.pkl", 'wb') as f:
            pickle.dump(a, f)
        os.remove("t.pkl")

    def test_condition_on_one_param_affects_other_even_if_cond_isnt_written(self):
        combs = MockMetaModule_with_implicit_condition.get_all_hyperparameter_combinations()
        to_tpl = lambda d: tuple(d.items())
        self.assertSetEqual({to_tpl(dict(param1='a', param1_a__param2=1, param2_1__param3=3)),
                             to_tpl(dict(param1='a', param1_a__param2=1, param2_1__param3=4)),
                             to_tpl(dict(param1='a', param1_a__param2=2)),
                             to_tpl(dict(param1='b'))}, set([to_tpl(d) for d in combs]))