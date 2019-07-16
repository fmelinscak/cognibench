import numpy as np
import gym
from gym import spaces
from scipy import stats
from .base import CAMO
from ...capabilities import Interactive, LogProbModel

class RandomRespondModel(CAMO, Interactive, LogProbModel):
    name = 'RandomRespond'

    def __init__(self, *args, mu, sigma, **kwargs):
        paras = dict(mu=mu, sigma=sigma)
        super().__init__(paras=paras, **kwargs)

    def reset(self):
        self.hidden_state = dict()

    def observation(self, stimulus):
        assert self.observation_space.contains(stimulus)

        mu_pred = self.paras['mu']
        sd_pred = self.paras['sigma']
        
        rv = stats.norm(loc=mu_pred, scale=sd_pred)
        rv.random_state = self.seed

        return rv

    def predict(self, stimulus):
        return self.observation(stimulus).logpdf

    def act(self, stimulus):
        return self.observation(stimulus).rvs()

    def update(self, stimulus, reward, action, done):
        assert self.action_space.contains(action)
        assert self.observation_space.contains(stimulus)
