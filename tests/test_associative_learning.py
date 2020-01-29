import unittest
from gym import spaces
import numpy as np
import numpy.testing as npt
from scipy import stats
from cognibench.models import associative_learning
from cognibench.continuous import ContinuousSpace


class Test_RwNormModel(unittest.TestCase):
    def setUp(self):
        # load test data
        n_obs = 3
        paras = {
            "w": 0.1 * np.ones(n_obs),
            "eta": 0.5,
            "sigma": 0.5,
            "b0": 0.5,
            "b1": 0.5 * np.ones(n_obs),
        }
        self.agent = associative_learning.RwNormAgent(n_obs=3, paras_dict=paras)

    def test_action_space(self):
        self.assertIsInstance(self.agent.get_action_space(), ContinuousSpace)

    def test_observation_space(self):
        self.assertEqual(self.agent.get_observation_space(), spaces.MultiBinary(3))

    def test_update(self):
        self.agent.update(np.array([0, 1, 0], dtype=np.int8), 1, 1, False)
        w = np.array([0.1, 0.55, 0.1], dtype=np.float64)
        self.assertIsNone(
            npt.assert_almost_equal(self.agent.get_hidden_state()["w"], w)
        )

    def test_reset(self):
        self.agent.reset()
        w = np.array([0.1, 0.1, 0.1], dtype=np.float64)
        self.assertIsNone(
            npt.assert_almost_equal(self.agent.get_hidden_state()["w"], w)
        )


class Test_KrwNormModel(unittest.TestCase):
    def setUp(self):
        # load test data
        n_obs = 3
        paras = {
            "w": 0.1 * np.ones(n_obs),
            "sigma": 0.5,
            "b0": 0.5,
            "b1": 0.5 * np.ones(n_obs),
            "sigmaWInit": 0.5,
            "tauSq": 0.5,
            "sigmaRSq": 0.5,
        }
        self.agent = associative_learning.KrwNormAgent(n_obs=n_obs, paras_dict=paras)

    def test_update(self):
        stimulus = np.array([0, 1, 0], dtype=np.int8)
        self.agent.update(stimulus, 1, 1, False)
        w = np.array([0.1, 0.7, 0.1], dtype=np.float64)
        self.assertIsNone(
            npt.assert_almost_equal(self.agent.get_hidden_state()["w"], w)
        )

        C_pred = 0.5 * np.identity(3) + 0.5 * np.identity(3)
        K = C_pred.dot(stimulus) / (stimulus.dot(C_pred.dot(stimulus)) + 0.5)
        C = C_pred - K * stimulus * C_pred
        self.assertIsNone(
            npt.assert_almost_equal(self.agent.get_hidden_state()["C"], C)
        )

    def test_reset(self):
        self.agent.reset()
        w = np.array([0.1, 0.1, 0.1], dtype=np.float64)
        self.assertIsNone(
            npt.assert_almost_equal(self.agent.get_hidden_state()["w"], w)
        )
        C = 0.5 * np.identity(3)
        self.assertIsNone(
            npt.assert_almost_equal(self.agent.get_hidden_state()["C"], C)
        )


class Test_LSSPDAgent(unittest.TestCase):
    def setUp(self):
        # load test data
        n_obs = 3
        paras = {
            "w": 0.1 * np.ones(n_obs),
            "alpha": 0.5 * np.ones(n_obs),
            "b0": 0.5,
            "b1": 0.5 * np.ones(n_obs),
            "mix_coef": 1,
            "sigma": 0.25,
            "eta": 0.3,
            "kappa": 0.4,
        }
        self.agent = associative_learning.LSSPDAgent(n_obs=n_obs, paras_dict=paras)

    def test_update(self):
        stimulus = np.array([0, 1, 0], dtype=np.int8)
        self.agent.update(stimulus, 1, 1, False)
        w = np.array([0.1, 0.28, 0.1], dtype=np.float64)
        self.assertIsNone(
            npt.assert_almost_equal(self.agent.get_hidden_state()["w"], w)
        )

        alpha = np.array([0.5, 0.62, 0.5], dtype=np.float64)
        self.assertIsNone(
            npt.assert_almost_equal(self.agent.get_hidden_state()["alpha"], alpha)
        )

    def test_reset(self):
        self.agent.reset()
        w = np.array([0.1, 0.1, 0.1], dtype=np.float64)
        self.assertIsNone(
            npt.assert_almost_equal(self.agent.get_hidden_state()["w"], w)
        )

        alpha = np.array([0.5, 0.5, 0.5], dtype=np.float64)
        self.assertIsNone(
            npt.assert_almost_equal(self.agent.get_hidden_state()["alpha"], alpha)
        )


class Test_BetaBinomialAgent(unittest.TestCase):
    def setUp(self):
        paras = {
            "intercept": 0.5,
            "slope": 0.5,
            "mix_coef": 1,
            "sigma": 0.5,
            "a": 1,
            "b": 1,
        }
        self.agent = associative_learning.BetaBinomialAgent(n_obs=3, paras_dict=paras)

    def test_update(self):
        stimulus = np.array([0, 1, 0], dtype=np.int8)
        self.agent.update(stimulus, 1, 1, False)
        a = np.array([1, 2, 1], dtype=np.float64)
        self.assertIsNone(
            npt.assert_almost_equal(self.agent.get_hidden_state()[stimulus]["a"], a)
        )

        b = np.array([1, 1, 1], dtype=np.float64)
        self.assertIsNone(
            npt.assert_almost_equal(self.agent.get_hidden_state()[stimulus]["b"], b)
        )


if __name__ == "__main__":
    unittest.main()
