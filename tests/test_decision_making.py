import unittest
from src.ldmunit.models import decision_making
from gym import spaces
import numpy as np
from scipy import stats

class Test_TestRWCKModel(unittest.TestCase):
    
    def setUp(self):
        # load test data
        paras = {'w0': 0.1, 'alpha': 0.5, 'alpha_c': 0.5, 'beta': 0.5, 'beta_c': 0.5}
        self.model = decision_making.RWCKModel(3, 3, paras)

    def test_action_space(self):
        self.assertIsInstance(self.model.action_space, spaces.Discrete)

    def test_observation_space(self):
        self.assertIsInstance(self.model.observation_space, spaces.Discrete)

    # def test_update(self):
    #     CK, Q = np.array([0.  , 0.75, 0.  ]), np.array([0.1  , 0.775, 0.1  ])
    #     res = self.model.update(0, 1, 1, False)
    #     self.model.reset()
    #     self.assertTupleEqual(res, (CK, Q))

if __name__ == '__main__':
    unittest.main()