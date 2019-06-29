import sciunit
import numpy as np
import gym
from gym import spaces
from scipy import stats
from .base import CAMO

class RandomRespondModel(CAMO):

    def __init__(self, n_obs=None, paras=None, hidden_state=None, name=None, **params):
        return super().__init__(n_obs=n_obs, paras=paras, hidden_state=hidden_state, name=name, **params)

    def _get_default_paras(self):
        return dict(mu=0, sigma=1)

    def reset(self):
        self.hidden_state = None

    def observation(self, stimulus):
        assert isinstance(self.observation_space, spaces.MultiBinary), "observation space must be set first"
        assert self.observation_space.contains(stimulus)

        mu_pred = self.paras['mu']
        sd_pred = self.paras['sigma']

        return stats.norm(loc=mu_pred, scale=sd_pred)

    def predict(self, stimulus):
        return self.observation(stimulus).logpdf

    def act(self, stimulus):
        return self.observation(stimulus).rvs()

    def update(self, stimulus, reward, action, done):
        assert self.action_space.contains(action)
        assert self.observation_space.contains(stimulus)

        pass

