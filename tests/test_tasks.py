import unittest
from gym import spaces
from functools import reduce
import numpy as np
from scipy import stats
from cognibench.models import associative_learning
from cognibench.envs import BanditEnv, ClassicalConditioningEnv
from cognibench.utils import partialclass
from cognibench.tasks import model_recovery, param_recovery
from cognibench.testing import InteractiveTest
from cognibench.scores import NLLScore


class TestSingleSubjectModelRecovery(unittest.TestCase):
    def setUp(self):
        # models = [
        #    associative_learning.RwNormModel,
        #    associative_learning.KrwNormModel,
        #    associative_learning.BetaBinomialModel,
        #    associative_learning.LSSPDModel,
        # ]
        # env_stimuli = [
        #    (np.array([0, 0, 0, 0, 0]), 0.1, 0.3),
        #    (np.array([0, 1, 0, 0, 1]), 0.4, 0.5),
        #    (np.array([0, 1, 1, 1, 0]), 0.25, 0.8),
        #    (np.array([1, 1, 1, 1, 1]), 0.25, 0.45),
        # ]

        # self.model_list = []
        # for ctor in models:
        #    model = ctor(n_obs=5)
        #    model.init_paras()
        #    self.model_list.append(model)

        # self.stimuli, self.p_stimuli, self.p_reward = zip(*env_stimuli)
        # env = ClassicalConditioningEnv(
        #    stimuli=self.stimuli, p_stimuli=self.p_stimuli, p_reward=self.p_reward
        # )
        # test_cls = partialclass(
        #    InteractiveTest,
        #    multi_subject=False,
        #    score_type=partialclass(NLLScore, min_score=0, max_score=1e4),
        # )
        # self.suite, self.sm = model_recovery(
        #    self.model_list, env, test_cls, n_trials=5, seed=42
        # )
        pass

    def test_(self):
        # TODO
        pass


class TestMultiSubjectModelRecovery(unittest.TestCase):
    def setUp(self):
        # TODO
        pass

    def test_(self):
        # TODO
        pass


class TestParamRecovery(unittest.TestCase):
    def setUp(self):
        # TODO
        pass

    def test_(self):
        # TODO
        pass


if __name__ == "__main__":
    unittest.main()
