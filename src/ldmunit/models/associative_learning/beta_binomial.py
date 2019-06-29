import sciunit
import numpy as np
import gym
from gym import spaces
from scipy import stats
from scipy.stats import beta
from .base import CAMO

class BetaBinomialModel(CAMO):

    def __init__(self, n_obs=None, paras=None, hidden_state=None, name=None, **params):
        return super().__init__(n_obs=n_obs, paras=paras, hidden_state=hidden_state, name=name, **params)

    def _set_a_b(self):
        assert isinstance(self.n_obs, int), "observation space must be set"
        return dict(a=np.ones(self.n_obs, dtype=np.float64), b=np.ones(self.n_obs, dtype=np.float64))

    def reset(self):
        assert isinstance(self.n_obs, int), "observation space must be set"
        self.hidden_state = dict()

    def observation(self, stimulus):
        assert self.observation_space.contains(stimulus)
        # get model's state
        if stimulus not in self.hidden_state.keys():
            self.hidden_state[stimulus] = self._set_a_b()
        a = self.hidden_state[stimulus]['a']
        b = self.hidden_state[stimulus]['b']

        rv = beta(a, b)
        return rv

    def predict(self, stimulus):
        return self.observation(stimulus).logpdf

    def act(self, stimulus):
        return self.observation(stimulus).rvs()

    def update(self, stimulus, reward, action, done): #TODO: add default value
        assert self.observation_space.contains(stimulus)
        # get model's state
        if stimulus not in self.hidden_state.keys():
            self.hidden_state[stimulus] = self._set_a_b()
        a = self.hidden_state[stimulus]['a']
        b = self.hidden_state[stimulus]['b']

        if not done:
            a += (1 - stimulus) * (1 - reward) + stimulus * reward
            b += (1 - stimulus) * reward + stimulus * (1 - reward)
            self.hidden_state[stimulus]['a'] = a
            self.hidden_state[stimulus]['b'] = b

        return a, b

    def act(self, stimulus):
        """observation function"""
        assert self.observation_space.contains(stimulus)

        mix_coef = self.paras['mix_coef']
        b0 = self.paras['b0'] # intercept
        b1 = self.paras['b1'] # slope #TODO: add variable slopes

        if stimulus not in self.hidden_state.keys():
            self.hidden_state[stimulus] = self._set_a_b()
        a = self.hidden_state[stimulus]['a']
        b = self.hidden_state[stimulus]['b']

        # Generate outcome prediction
        mu      = beta.mean(a, b)
        entropy = beta.entropy(a, b)

        crPred = b0 + np.dot(stimulus, (mix_coef * mu  + (1 - mix_coef) * entropy)) * b1
        
        return np.array([crPred])
