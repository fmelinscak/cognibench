import unittest
from gym import spaces
import numpy as np
import numpy.testing as npt
from scipy import stats
from src.ldmunit.models import decision_making

class Test_RWCKModel(unittest.TestCase):
    
    def setUp(self):
        # load test data
        paras = {'w0': 0.1, 'alpha': 0.5, 'alpha_c': 0.5, 'beta': 0.5, 'beta_c': 0.5}
        self.model = decision_making.RWCKModel(3, 3, **paras)

    def test_empty_init(self):
        self.model = decision_making.RWCKModel()
        self.assertIsNone(self.model.paras)
        self.assertIs(self.model.action_space, spaces.Discrete)
        self.assertIs(self.model.observation_space, spaces.Discrete)
        self.assertIsNone(self.model.n_action)
        self.assertIsNone(self.model.n_obs)
        self.setUp()

    def test_action_space(self):
        self.assertEqual(self.model.action_space, spaces.Discrete(3))

    def test_observation_space(self):
        self.assertEqual(self.model.observation_space, spaces.Discrete(3))

    def test_update(self):
        self.model.update(0, 1, 0, False)
        CK = {0:np.array([0.5 , 0  , 0  ]), 1: np.zeros(3    ), 2: np.zeros(3     )}
        Q  = {0:np.array([0.55, 0.1, 0.1]), 1: np.full (3,0.1), 2: np.full (3, 0.1)}
        for i in range(3):
            self.assertIsNone(npt.assert_almost_equal(self.model.hidden_state['CK'][i], CK[i]))
            self.assertIsNone(npt.assert_almost_equal(self.model.hidden_state['Q'][i], Q[i]))

    def test_reset(self):
        self.model.reset()
        CK = {0:np.zeros (3    ), 1: np.zeros (3    ), 2: np.zeros (3     )}
        Q  = {0:np.full  (3,0.1), 1: np.full  (3,0.1), 2: np.full  (3, 0.1)}
        for i in range(3):
            self.assertIsNone(npt.assert_almost_equal(self.model.hidden_state['CK'][i], CK[i]))
            self.assertIsNone(npt.assert_almost_equal(self.model.hidden_state['Q'][i], Q[i]))

    def test__get_rv(self):
        self.assertIsInstance(self.model._get_rv(0), stats.rv_discrete)


class Test_NWSLSModel(unittest.TestCase):
    
    def setUp(self):
        # load test data
        paras = {'epsilon': 0.5}
        self.model = decision_making.NWSLSModel(3, 3, **paras)

    def test_empty_init(self):
        self.model = decision_making.NWSLSModel()
        self.assertIsNone(self.model.paras)
        self.assertIs(self.model.action_space, spaces.Discrete)
        self.assertIs(self.model.observation_space, spaces.Discrete)
        self.assertIsNone(self.model.n_action)
        self.assertIsNone(self.model.n_obs)
        self.setUp()
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
        self.model = decision_making.RandomRespondModel(3, 3, **paras)

    def test_empty_init(self):
        self.model = decision_making.RandomRespondModel()
        self.assertIsNone(self.model.paras)
        self.assertIs(self.model.action_space, spaces.Discrete)
        self.assertIs(self.model.observation_space, spaces.Discrete)
        self.assertIsNone(self.model.n_action)
        self.assertIsNone(self.model.n_obs)
        self.setUp()
    def test_action_space(self):
        self.assertEqual(self.model.action_space, spaces.Discrete(3))

    def test_observation_space(self):
        self.assertEqual(self.model.observation_space, spaces.Discrete(3))
    
    def test_update(self):
        self.assertIsNone(self.model.update(0, 1, 1, False))

    def test_reset(self):
        self.model.reset()
        self.assertEqual(1, self.model.hidden_state)
  
    def test__get_rv(self):
        self.assertIsInstance(self.model._get_rv(0), stats.rv_discrete)

if __name__ == '__main__':
    unittest.main()
