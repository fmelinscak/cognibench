import unittest
from gym import spaces
from functools import reduce
import numpy as np
import numpy.testing as npt
from scipy import stats
from cognibench.models import decision_making
from cognibench.envs import BanditEnv, ClassicalConditioningEnv
from cognibench.simulation import simulate
from cognibench.capabilities import (
    DiscreteAction,
    DiscreteObservation,
    ContinuousAction,
    MultiBinaryObservation,
)


class TestBanditEnv(unittest.TestCase):
    def setUp(self):
        self.p_dist = [0.1, 0.3, 0.4, 0.2]
        self.env = BanditEnv(p_dist=self.p_dist, seed=42)

    def test_structure(self):
        self.assertEqual(self.env.n_bandits, len(self.p_dist))
        self.assertTrue(self.env.get_action_space(), DiscreteAction)
        self.assertTrue(self.env.get_observation_space(), DiscreteObservation)

    def test_methods(self):
        self.assertRaises(AssertionError, self.env.step, -1)
        self.assertRaises(AssertionError, self.env.step, len(self.p_dist) + 1)
        self.assertTrue(0 <= self.env.reset() < len(self.p_dist))


class TestClassicalConditioningEnv(unittest.TestCase):
    def setUp(self):
        env_stimuli = [
            (np.array([0, 0, 0, 0, 0]), 0.1, 0.3),
            (np.array([0, 1, 0, 0, 1]), 0.4, 0.5),
            (np.array([0, 1, 1, 1, 0]), 0.25, 0.8),
            (np.array([1, 1, 1, 1, 1]), 0.25, 0.45),
        ]
        self.stimuli, self.p_stimuli, self.p_reward = zip(*env_stimuli)
        self.env = ClassicalConditioningEnv(
            stimuli=self.stimuli, p_stimuli=self.p_stimuli, p_reward=self.p_reward
        )

    def test_structure(self):
        self.assertTrue(self.env.get_action_space(), ContinuousAction)
        self.assertTrue(self.env.get_observation_space(), MultiBinaryObservation)

    def test_methods(self):
        self.assertRaises(AssertionError, self.env.step, [0, 1, 1, 1, 1])
        self.assertRaises(AssertionError, self.env.step, [-1, -1, -1, -1, -1])

        reset_in = False
        reset_stim = self.env.reset()
        for s in self.stimuli:
            reset_in = reset_in or (reset_stim == s).all()
        self.assertTrue(reset_in)


if __name__ == "__main__":
    unittest.main()
