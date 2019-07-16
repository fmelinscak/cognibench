import unittest
from gym import spaces
import numpy as np
import numpy.testing as npt
from scipy import stats
from src.ldmunit.models import associative_learning
from src.ldmunit.continuous import Continuous

class Test_RwNormModel(unittest.TestCase):
    
    def setUp(self):
        # load test data
        paras = {'w0': 0.1, 'alpha': 0.5, 'sigma': 0.5, 'b0': 0.5, 'b1': 0.5}
        self.model = associative_learning.RwNormModel(3, **paras)

    def test_action_space(self):
        self.assertIsInstance(self.model.action_space, Continuous)

    def test_observation_space(self):
        self.assertEqual(self.model.observation_space, spaces.MultiBinary(3))

    def test_update(self):
        self.model.update(np.array([0,1,0], dtype=np.int8), 1, 1, False)
        w = np.array([0.1, 0.55, 0.1], dtype=np.float64)
        self.assertIsNone(npt.assert_almost_equal(self.model.hidden_state['w'], w))

    def test_reset(self):
        self.model.reset()
        w = np.array([0.1, 0.1, 0.1], dtype=np.float64)
        self.assertIsNone(npt.assert_almost_equal(self.model.hidden_state['w'], w))

class Test_KrwNormModel(unittest.TestCase):
    
    def setUp(self):
        # load test data
        paras = {'w0': 0.1, 'sigma': 0.5, 'b0': 0.5, 'b1': 0.5, 'logSigmaWInit': 0.5, 'logTauSq': 0.5, 'logSigmaRSq': 0.5}
        self.model = associative_learning.KrwNormModel(3, **paras)

    def test_update(self):
        stimulus = np.array([0,1,0], dtype=np.int8)
        self.model.update(stimulus, 1, 1, False)
        w = np.array([0.1, 0.7, 0.1], dtype=np.float64)
        self.assertIsNone(npt.assert_almost_equal(self.model.hidden_state['w'], w))

        C_pred = np.exp(0.5) * np.identity(3) + np.exp(0.5) * np.identity(3)
        K = C_pred.dot(stimulus) / (stimulus.dot(C_pred.dot(stimulus)) + np.exp(0.5))
        C = C_pred - K * stimulus * C_pred
        self.assertIsNone(npt.assert_almost_equal(self.model.hidden_state['C'], C))

    def test_reset(self):
        self.model.reset()
        w = np.array([0.1, 0.1, 0.1], dtype=np.float64)
        self.assertIsNone(npt.assert_almost_equal(self.model.hidden_state['w'], w))
        C = np.exp(0.5) * np.identity(3)
        self.assertIsNone(npt.assert_almost_equal(self.model.hidden_state['C'], C))


class Test_LSSPDModel(unittest.TestCase):
    
    def setUp(self):
        # load test data
        paras = {'w0': 0.1, 'alpha': 0.5, 'b0': 0.5, 'b1': 0.5, 'mix_coef': 1, 'eta': 0.3, 'kappa': 0.4}
        self.model = associative_learning.LSSPDModel(3, **paras)

    def test_update(self):
        stimulus = np.array([0,1,0], dtype=np.int8)
        self.model.update(stimulus, 1, 1, False)
        w = np.array([0.1, 0.28, 0.1], dtype=np.float64)
        self.assertIsNone(npt.assert_almost_equal(self.model.hidden_state['w'], w))

        alpha = np.array([0.5, 0.62, 0.5], dtype=np.float64)
        self.assertIsNone(npt.assert_almost_equal(self.model.hidden_state['alpha'], alpha))

    def test_reset(self):
        self.model.reset()
        w = np.array([0.1, 0.1, 0.1], dtype=np.float64)
        self.assertIsNone(npt.assert_almost_equal(self.model.hidden_state['w'], w))

        alpha = np.array([0.5, 0.5, 0.5], dtype=np.float64)
        self.assertIsNone(npt.assert_almost_equal(self.model.hidden_state['alpha'], alpha))

class Test_LSSPDModel(unittest.TestCase):
    
    def setUp(self):
        # load test data
        paras = {'w0': 0.1, 'alpha': 0.5, 'b0': 0.5, 'b1': 0.5, 'mix_coef': 1, 'eta': 0.3, 'kappa': 0.4}
        self.model = associative_learning.LSSPDModel(3, **paras)

    def test_update(self):
        stimulus = np.array([0,1,0], dtype=np.int8)
        self.model.update(stimulus, 1, 1, False)
        w = np.array([0.1, 0.28, 0.1], dtype=np.float64)
        self.assertIsNone(npt.assert_almost_equal(self.model.hidden_state['w'], w))

        alpha = np.array([0.5, 0.62, 0.5], dtype=np.float64)
        self.assertIsNone(npt.assert_almost_equal(self.model.hidden_state['alpha'], alpha))

    def test_reset(self):
        self.model.reset()
        w = np.array([0.1, 0.1, 0.1], dtype=np.float64)
        self.assertIsNone(npt.assert_almost_equal(self.model.hidden_state['w'], w))

        alpha = np.array([0.5, 0.5, 0.5], dtype=np.float64)
        self.assertIsNone(npt.assert_almost_equal(self.model.hidden_state['alpha'], alpha))

class Test_BetaBinomialModel(unittest.TestCase):

    def setUp(self):
        # load test data
        paras = {'b0': 0.5, 'b1': 0.5, 'mix_coef': 1, 'sigma': 0.5}
        self.model = associative_learning.BetaBinomialModel(3, **paras)

    def test_update(self):
        stimulus = np.array([0,1,0], dtype=np.int8)
        self.model.update(stimulus, 1, 1, False)
        a = np.array([1, 2, 1], dtype=np.float64)
        self.assertIsNone(npt.assert_almost_equal(self.model.hidden_state[stimulus]['a'], a))

        b = np.array([2, 1, 2], dtype=np.float64)
        self.assertIsNone(npt.assert_almost_equal(self.model.hidden_state[stimulus]['b'], b))

if __name__ == '__main__':
    unittest.main()
