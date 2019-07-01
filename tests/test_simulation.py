import unittest
from gym import spaces
import numpy as np
import numpy.testing as npt
from scipy import stats
from src.ldmunit.models import decision_making
from src.ldmunit.bandit import BanditEnv
from src.ldmunit.models.utils import simulate

class Test_Unit(unittest.TestCase):
    def setUp(self):
        self.env = BanditEnv([0.01, 0.99])
        paras = {'w0': 0., 'alpha': 1, 'alpha_c': 1, 'beta': 1, 'beta_c': 1}
        self.model = decision_making.RWCKModel(2, 2, paras=paras)

    def test_simulation(self):
        stimuli, rewards, actions = simulate(self.env, self.model, 100)
        self.assertGreater(self.model._get_rv(0).pk[1], 0.65)
        self.assertGreater(np.unique(actions, return_counts=True)[1][1], 65)