import unittest
from gym import spaces
import numpy as np
import numpy.testing as npt
from scipy import stats
from src.ldmunit.models import decision_making
from src.ldmunit.env import BanditEnv
from src.ldmunit.models.utils import multi_from_single_interactive, simulate_multi_env_multi_model

class Test_Unit(unittest.TestCase):
    def setUp(self):
        self.env = [BanditEnv([0.01, 0.99])]
        paras = [{'w': 0., 'eta': 1, 'eta_c': 1, 'beta': 1, 'beta_c': 1}]
        ModelClass = multi_from_single_interactive(decision_making.RWCKModel)
        self.model = ModelClass(paras, n_action=2, n_obs=2)

    def test_simulation(self):
        stimuli, rewards, actions = simulate_multi_env_multi_model(self.env, self.model, 100)
        self.assertGreater(self.model.subject_models[0]._get_rv(0).pk[1], 0.65)
        self.assertGreater(np.unique(actions[0], return_counts=True)[1][1], 65)

if __name__ == '__main__':
    unittest.main()
