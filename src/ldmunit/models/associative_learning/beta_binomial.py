import sciunit
import numpy as np
import gym
from gym import spaces
from scipy import stats
from scipy.stats import beta
from .base import CAMO

class NpDict(dict):
    "Automatically set np.ndarray to tuple for key access."
    def __getitem__(self, key):
        key = tuple(key) if isinstance(key, np.ndarray) else key
        val = dict.__getitem__(self, key)
        return val

    def __setitem__(self, key, val):
        key = tuple(key) if isinstance(key, np.ndarray) else key
        dict.__setitem__(self, key, val)

class BetaBinomialModel(CAMO):

    def __init__(self, n_obs=None, paras=None, hidden_state=None, name=None, **params):
        return super().__init__(n_obs=n_obs, paras=paras, hidden_state=hidden_state, name=name, **params)

    def _get_default_paras(self):
        return {'b0': 0.5, 'b1': 0.5, 'mix_coef': 1, 'sigma': 0.5}
 
    def _set_a_b(self):
        assert isinstance(self.n_obs, int), "observation space must be set"
        return dict(a=np.ones(self.n_obs, dtype=np.float64), b=np.ones(self.n_obs, dtype=np.float64))

    def reset(self):
        assert isinstance(self.n_obs, int), "observation space must be set"
        self.hidden_state = NpDict()

    def observation(self, stimulus):
        assert isinstance(self.observation_space, spaces.MultiBinary), "observation space must be set first"
        assert self.hidden_state, "hidden state must be set"
        assert self.observation_space.contains(stimulus)

        sd_pred = self.paras['sigma']
        
        mu_pred = self._predict_reward(stimulus)

        return stats.norm(loc=mu_pred, scale=sd_pred)

    def predict(self, stimulus):
        if tuple(stimulus) not in self.hidden_state.keys():
            self.hidden_state[stimulus] = self._set_a_b()
        return self.observation(stimulus).logpdf

    def act(self, stimulus):
        if tuple(stimulus) not in self.hidden_state.keys():
            self.hidden_state[stimulus] = self._set_a_b()
        return self.observation(stimulus).rvs()

    def update(self, stimulus, reward, action, done): #TODO: add default value
        assert self.observation_space.contains(stimulus)
        # get model's state
        if tuple(stimulus) not in self.hidden_state.keys():
            self.hidden_state[stimulus] = self._set_a_b()
        a = self.hidden_state[stimulus]['a']
        b = self.hidden_state[stimulus]['b']

        if not done:
            a += (1 - stimulus) * (1 - reward) + stimulus * reward
            b += (1 - stimulus) * reward + stimulus * (1 - reward)
            self.hidden_state[stimulus]['a'] = a
            self.hidden_state[stimulus]['b'] = b

        return a, b

    def _predict_reward(self, stimulus):
        mix_coef = self.paras['mix_coef']
        b0 = self.paras['b0'] # intercept
        b1 = self.paras['b1'] # slope #TODO: add variable slopes

        if tuple(stimulus) not in self.hidden_state.keys():
            self.hidden_state[stimulus] = self._set_a_b()
        a = self.hidden_state[stimulus]['a']
        b = self.hidden_state[stimulus]['b']

        mu      = beta(a,b).mean()
        entropy = beta(a,b).entropy()

        rhat = b0 + np.dot(stimulus, (mix_coef * mu  + (1 - mix_coef) * entropy)) * b1
        
        return rhat
