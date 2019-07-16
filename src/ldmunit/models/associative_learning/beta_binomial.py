import sciunit
import numpy as np
import gym
from gym import spaces
from scipy import stats
from scipy.stats import beta
from .base import CAMO
from ...capabilities import Interactive, LogProbModel

class BetaBinomialModel(CAMO, Interactive, LogProbModel):
    name = 'BetaBinomial'

    def __init__(self, n_obs=None, paras=None, hidden_state=None, name=None, seed=None, **params):
        return super().__init__(n_obs=n_obs, paras=paras, hidden_state=hidden_state, name=name, seed=seed, **params)

    def _set_a_b(self):
        assert isinstance(self.n_obs, int), "observation space must be set"
        a = np.float(self.paras['aInit']) if 'aInit' in self.paras else 1.
        b = np.float(self.paras['bInit']) if 'aInit' in self.paras else 1.
        return dict(a=np.ones(self.n_obs, dtype=np.float64), b=np.ones(self.n_obs, dtype=np.float64))

    def reset(self, paras=None):
        assert isinstance(self.n_obs, int), "observation space must be set"
        self.hidden_state = dict()

    def observation(self, stimulus, paras=None):
        if not paras:
            paras = self.paras
        assert isinstance(self.observation_space, spaces.MultiBinary), "observation space must be set first"
        assert self.hidden_state, "hidden state must be set"
        assert self.observation_space.contains(stimulus)

        sd_pred = paras['sigma']
        
        mu_pred = self._predict_reward(stimulus)

        rv = stats.norm(loc=mu_pred, scale=sd_pred)
        if self.seed:
            rv.random_state = self.seed

        return rv


    def predict(self, stimulus, paras=None):
        if tuple(stimulus) not in self.hidden_state.keys():
            self._set_hidden_state(stimulus, self._set_a_b())
        return self.observation(stimulus, paras).logpdf

    def act(self, stimulus, paras=None):
        if tuple(stimulus) not in self.hidden_state.keys():
            self._set_hidden_state(stimulus, self._set_a_b())
        return self.observation(stimulus, paras).rvs()

    def update(self, stimulus, reward, action, done, paras=None):
        assert self.observation_space.contains(stimulus)
        # get model's state
        if tuple(stimulus) not in self.hidden_state.keys():
            self._set_hidden_state(stimulus, self._set_a_b())
        a = self._get_hidden_state(stimulus)['a']
        b = self._get_hidden_state(stimulus)['b']

        if not done:
            a += (1 - stimulus) * (1 - reward) + stimulus * reward
            b += (1 - stimulus) * reward + stimulus * (1 - reward)
            old = self._get_hidden_state(stimulus)
            old['a'] = a
            old['b'] = b
            self._set_hidden_state(stimulus, old)

        return a, b

    def _set_hidden_state(self, key, val):
        self.hidden_state[tuple(key)] = val

    def _get_hidden_state(self, key):
        return self.hidden_state[tuple(key)]

    def _predict_reward(self, stimulus, paras=None):
        if not paras:
            paras = self.paras
        mix_coef = paras['mix_coef']
        b0 = paras['b0'] # intercept
        b1 = paras['b1'] # slope

        if tuple(stimulus) not in self.hidden_state.keys():
            self._set_hidden_state(stimulus, self._set_a_b())
        a = self._get_hidden_state(stimulus)['a']
        b = self._get_hidden_state(stimulus)['b']

        mu      = beta(a,b).mean()
        entropy = beta(a,b).entropy()

        rhat = b0 + np.dot(stimulus, (mix_coef * mu  + (1 - mix_coef) * entropy)) * b1
        
        return rhat
