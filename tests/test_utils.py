import unittest
from gym import spaces
from functools import reduce
import numpy as np
from scipy import stats
from ldmunit.models import associative_learning
from ldmunit.models.utils import (
    multi_from_single_cls,
    single_from_multi_obj,
    reverse_single_from_multi_obj,
)
from ldmunit.envs import BanditEnv, ClassicalConditioningEnv
from ldmunit.utils import partialclass, negloglike, is_arraylike
from ldmunit.tasks import model_recovery, param_recovery
from ldmunit.testing import InteractiveTest
from ldmunit.scores import NLLScore


class Test_partialclass(unittest.TestCase):
    class Complex:
        def __init__(self, *args, x, y, name):
            self.x = x
            self.y = y
            self.name = name

    def setUp(self):
        self.partial_init_with_x = partialclass(Test_partialclass.Complex, x=1)
        self.partial_init_with_y = partialclass(Test_partialclass.Complex, y=2)
        self.partial_init_with_name = partialclass(Test_partialclass.Complex, name="c1")
        self.partial_init_with_xy = partialclass(Test_partialclass.Complex, x=1, y=2)

    def test_same_object(self):
        obj_list = [
            self.partial_init_with_x(y=2, name="c1"),
            self.partial_init_with_y(x=1, name="c1"),
            self.partial_init_with_name(x=1, y=2),
            self.partial_init_with_xy(name="c1"),
        ]
        for i in range(len(obj_list) - 1):
            self.assertEqual(obj_list[i].x, obj_list[i + 1].x)
            self.assertEqual(obj_list[i].y, obj_list[i + 1].y)
            self.assertEqual(obj_list[i].name, obj_list[i + 1].name)


class Test_negloglike(unittest.TestCase):
    distr = [0.1, 0.2, 0.3, 0.15, 0.25]

    def logpmf(rv):
        return np.log(Test_negloglike.distr[rv])

    def test_custom(self):
        actions_list = [
            [0, 0, 2, 3, 1, 1, 4],
            [0, 0, 0, 0],
            [0, 1, 2, 3, 4],
            [1, 1, 1, 1, 1],
        ]
        for actions in actions_list:
            expected = sum(-np.log(Test_negloglike.distr[i]) for i in actions)
            actual = negloglike(actions, [Test_negloglike.logpmf] * len(actions))
            self.assertAlmostEqual(expected, actual)


class Test_multi_from_single_cls(unittest.TestCase):
    def setUp(self):
        self.single_model_cls = associative_learning.KrwNormModel
        self.multi_model_cls = multi_from_single_cls(self.single_model_cls)

    def test_single_multi_equality(self):
        kwargs = {"n_obs": 3, "seed": 42}
        n_subj = 5
        single_obj = self.single_model_cls(**kwargs)
        multi_obj = self.multi_model_cls(n_subj=n_subj, **kwargs)
        for i in range(n_subj):
            single_paras = single_obj.get_paras()
            multi_paras = multi_obj.get_paras(i)
            for k, v in single_paras.items():
                if is_arraylike(v):
                    self.assertTrue((v == multi_paras[k]).all())
                else:
                    self.assertEqual(v, multi_paras[k])


class Test_single_from_multi_obj(unittest.TestCase):
    def setUp(self):
        self.multi_cls = multi_from_single_cls(associative_learning.KrwNormModel)

    def test_update_one_subject_model(self):
        n_subj = 5
        subj_ids = range(n_subj)
        multi_obj_0 = self.multi_cls(n_obs=3, n_subj=n_subj, seed=45)
        multi_obj_1 = self.multi_cls(n_obs=3, n_subj=n_subj, seed=45)
        for i in subj_ids:
            stimulus = [0, 1, 0]
            logpdf_native_multi = multi_obj_0.predict(i, stimulus)
            single_proxy = single_from_multi_obj(multi_obj_1, i)
            logpdf_proxy = single_proxy.predict(stimulus)
            multi_obj_1 = reverse_single_from_multi_obj(multi_obj_1)
            trial_pts = [-0.25, 0, 0.25, 0.5, 1, 5]
            for pt in trial_pts:
                self.assertEqual(logpdf_native_multi(pt), logpdf_proxy(pt))
