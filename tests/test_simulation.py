import unittest
from gym import spaces
import numpy as np
import numpy.testing as npt
from scipy import stats
from cognibench.models import decision_making
from cognibench.envs import BanditEnv
from cognibench.simulation import simulate


class Test_Unit(unittest.TestCase):
    def setUp(self):
        self.env = BanditEnv(p_dist=[0.15, 0.85])
        paras = {"w": 0.5, "eta": 1e-1, "eta_c": 1e-1, "beta": 2.5, "beta_c": 0.25}
        self.agent = decision_making.RWCKAgent(n_obs=1, n_action=2, paras_dict=paras)

    def test_simulation(self):
        stimuli, rewards, actions = simulate(self.env, self.agent, 100)
        self.assertEqual(self.agent.eval_policy(0).pk[1], 0.8757028699917109)
        self.assertEqual(np.unique(actions, return_counts=True)[1][1], 87)


if __name__ == "__main__":
    unittest.main()
