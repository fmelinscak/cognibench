import unittest
from gym import spaces
import numpy as np
import numpy.testing as npt
from cognibench.models import decision_making
from cognibench import distr


class Test_RWCKAgent(unittest.TestCase):
    def setUp(self):
        # load test data
        paras = {"w": 0.1, "eta": 0.5, "eta_c": 0.5, "beta": 0.5, "beta_c": 0.5}
        self.agent = decision_making.RWCKAgent(n_action=3, n_obs=3, paras_dict=paras)

    def test_action_space(self):
        self.assertEqual(self.agent.get_action_space(), spaces.Discrete(3))

    def test_observation_space(self):
        self.assertEqual(self.agent.get_observation_space(), spaces.Discrete(3))

    def test_update(self):
        self.agent.update(0, 1, 0, False)
        CK = np.c_[np.array([0.5, 0, 0]), np.zeros(3), np.zeros(3)]
        Q = np.c_[np.array([0.55, 0.1, 0.1]), np.full(3, 0.1), np.full(3, 0.1)]
        for i in range(3):
            self.assertIsNone(
                npt.assert_almost_equal(self.agent.get_hidden_state()["CK"][i], CK[i])
            )
            self.assertIsNone(
                npt.assert_almost_equal(self.agent.get_hidden_state()["Q"][i], Q[i])
            )

    def test_reset(self):
        self.agent.reset()
        CK = np.zeros((3, 3))
        Q = np.full((3, 3), 0.1)
        for i in range(3):
            self.assertIsNone(
                npt.assert_almost_equal(self.agent.get_hidden_state()["CK"][i], CK[i])
            )
            self.assertIsNone(
                npt.assert_almost_equal(self.agent.get_hidden_state()["Q"][i], Q[i])
            )

    def test_eval_policy(self):
        self.assertIsInstance(self.agent.eval_policy(0), distr.DiscreteRV)


class Test_NWSLSAgent(unittest.TestCase):
    def setUp(self):
        # load test data
        paras = {"epsilon": 0.5}
        self.agent = decision_making.NWSLSAgent(n_action=3, n_obs=3, paras_dict=paras)

    def test_action_space(self):
        self.assertEqual(self.agent.get_action_space(), spaces.Discrete(3))

    def test_observation_space(self):
        self.assertEqual(self.agent.get_observation_space(), spaces.Discrete(3))

    def test_update(self):
        self.agent.update(0, 1, 1, False)
        win = True
        action = 1
        self.assertEqual(win, self.agent.get_hidden_state()["win"])
        self.assertEqual(action, self.agent.get_hidden_state()["action"])

    def test_reset(self):
        self.agent.reset()
        win = True
        self.assertEqual(win, self.agent.get_hidden_state()["win"])
        self.assertIn(self.agent.get_hidden_state()["action"], [0, 1, 2])

    def test_eval_policy(self):
        self.assertIsInstance(self.agent.eval_policy(0), distr.DiscreteRV)


class Test_RandomRespondAgent(unittest.TestCase):
    def setUp(self):
        # load test data
        paras = {"bias": 0.5, "action_bias": 1}
        self.agent = decision_making.RandomRespondAgent(
            n_action=3, n_obs=3, paras_dict=paras
        )

    def test_action_space(self):
        self.assertEqual(self.agent.get_action_space(), spaces.Discrete(3))

    def test_observation_space(self):
        self.assertEqual(self.agent.get_observation_space(), spaces.Discrete(3))

    def test_update(self):
        self.assertIsNone(self.agent.update(0, 1, 1, False))

    def test_reset(self):
        self.agent.reset()
        self.assertEqual(dict(), self.agent.get_hidden_state())

    def test_eval_policy(self):
        self.assertIsInstance(self.agent.eval_policy(0), distr.DiscreteRV)


if __name__ == "__main__":
    unittest.main()
