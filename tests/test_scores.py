import unittest
from gym import spaces
from functools import reduce
import numpy as np
from scipy import stats
from ldmunit.models import associative_learning
from ldmunit.envs import BanditEnv, ClassicalConditioningEnv
from ldmunit.utils import partialclass
from ldmunit.tasks import model_recovery, param_recovery
from ldmunit.testing import InteractiveTest
from ldmunit.scores import NLLScore


class TestHigherBetterScore(unittest.TestCase):
    def setUp(self):
        pass

    def test_(self):
        # TODO
        pass


class TestLowerBetterScore(unittest.TestCase):
    def setUp(self):
        # TODO
        pass

    def test_(self):
        # TODO
        pass
