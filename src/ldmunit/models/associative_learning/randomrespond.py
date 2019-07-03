import sciunit
import numpy as np
import gym
from gym import spaces
from scipy import stats
from .base import CAMO

class RandomRespondModel(CAMO):

    def __init__(self, n_obs=None, paras=None, hidden_state=None, name=None, seed=None, **params):
        return super().__init__(n_obs=n_obs, paras=paras, hidden_state=hidden_state, name=name, seed=seed, **params)
        
    def reset(self, paras=None):
        self.hidden_state = None

    def observation(self, stimulus, paras=None):
        if not paras:
            paras = self.paras
        assert isinstance(self.observation_space, spaces.MultiBinary), "observation space must be set first"
        assert self.observation_space.contains(stimulus)

        mu_pred = paras['mu']
        sd_pred = paras['sigma']
        
        rv = stats.norm(loc=mu_pred, scale=sd_pred)
        if self.seed:
            rv.random_state = self.seed

        return rv

    def predict(self, stimulus):
        return self.observation(stimulus).logpdf

    def act(self, stimulus):
        return self.observation(stimulus).rvs()

    def update(self, stimulus, reward, action, done, paras=None):
        assert self.action_space.contains(action)
        assert self.observation_space.contains(stimulus)

        pass

