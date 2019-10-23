import unittest
from gym import spaces
import numpy as np
import numpy.testing as npt
from scipy import stats
from ldmunit.models import decision_making


class Test_RWCKModel(unittest.TestCase):
    def setUp(self):
        # load test data
        paras = {'w': 0.1, 'eta': 0.5, 'eta_c': 0.5, 'beta': 0.5, 'beta_c': 0.5}
        self.model = decision_making.RWCKModel(n_action=3, n_obs=3, **paras)

    def test_action_space(self):
        self.assertEqual(self.model.action_space, spaces.Discrete(3))

    def test_observation_space(self):
        self.assertEqual(self.model.observation_space, spaces.Discrete(3))

    def test_update(self):
        self.model.update(0, 1, 0, False)
        CK = np.c_[np.array([0.5, 0, 0]), np.zeros(3), np.zeros(3)]
        Q = np.c_[np.array([0.55, 0.1, 0.1]), np.full(3, 0.1), np.full(3, 0.1)]
        for i in range(3):
            self.assertIsNone(npt.assert_almost_equal(self.model.hidden_state['CK'][i], CK[i]))
            self.assertIsNone(npt.assert_almost_equal(self.model.hidden_state['Q'][i], Q[i]))

    def test_reset(self):
        self.model.reset()
        CK = np.zeros((3, 3))
        Q = np.full((3, 3), 0.1)
        for i in range(3):
            self.assertIsNone(npt.assert_almost_equal(self.model.hidden_state['CK'][i], CK[i]))
            self.assertIsNone(npt.assert_almost_equal(self.model.hidden_state['Q'][i], Q[i]))

    def test__get_rv(self):
        self.assertIsInstance(self.model._get_rv(0), stats.rv_discrete)


class Test_NWSLSModel(unittest.TestCase):
    def setUp(self):
        # load test data
        paras = {'epsilon': 0.5}
        self.model = decision_making.NWSLSModel(n_action=3, n_obs=3, **paras)

    def test_action_space(self):
        self.assertEqual(self.model.action_space, spaces.Discrete(3))

    def test_observation_space(self):
        self.assertEqual(self.model.observation_space, spaces.Discrete(3))

    def test_update(self):
        self.model.update(0, 1, 1, False)
        win = True
        action = 1
        self.assertEqual(win, self.model.hidden_state['win'])
        self.assertEqual(action, self.model.hidden_state['action'])

    def test_reset(self):
        self.model.reset()
        win = True
        self.assertEqual(win, self.model.hidden_state['win'])
        self.assertIn(self.model.hidden_state['action'], [0, 1, 2])

    def test__get_rv(self):
        self.assertIsInstance(self.model._get_rv(0), stats.rv_discrete)


class Test_RandomRespondModel(unittest.TestCase):
    def setUp(self):
        # load test data
        paras = {'bias': 0.5, 'action_bias': 1}
        self.model = decision_making.RandomRespondModel(n_action=3, n_obs=3, **paras)

    def test_action_space(self):
        self.assertEqual(self.model.action_space, spaces.Discrete(3))

    def test_observation_space(self):
        self.assertEqual(self.model.observation_space, spaces.Discrete(3))

    def test_update(self):
        self.assertIsNone(self.model.update(0, 1, 1, False))

    def test_reset(self):
        self.model.reset()
        self.assertEqual(dict(), self.model.hidden_state)

    def test__get_rv(self):
        self.assertIsInstance(self.model._get_rv(0), stats.rv_discrete)


if __name__ == '__main__':
    unittest.main()
