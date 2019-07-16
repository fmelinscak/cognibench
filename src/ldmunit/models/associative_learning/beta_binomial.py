import numpy as np
import gym
from gym import spaces
from scipy import stats
from scipy.stats import beta
from .base import CAMO
from ...capabilities import Interactive, LogProbModel

class BetaBinomialModel(CAMO, Interactive, LogProbModel):
    name = 'BetaBinomial'

    def __init__(self, *args, aInit=1, bInit=1, sigma, mix_coef, b0, b1, **kwargs):
        paras = {
            'aInit' : aInit,
            'bInit' : bInit,
            'sigma' : sigma,
            'mix_coef' : mix_coef,
            'b0' : b0,
            'b1' : b1
        }
        super().__init__(paras=paras, **kwargs)

    def _set_a_b(self):
        a = self.paras['aInit']
        b = self.paras['bInit']
        out = {'a' : a * np.ones(self.n_obs, dtype=np.float64),
                'b' : b * np.ones(self.n_obs, dtype=np.float64)}
        return out

    def reset(self):
        self.hidden_state = dict()

    def observation(self, stimulus):
        assert self.observation_space.contains(stimulus)

        sd_pred = self.paras['sigma']
        
        mu_pred = self._predict_reward(stimulus)

        rv = stats.norm(loc=mu_pred, scale=sd_pred)
        rv.random_state = self.seed

        return rv


    def predict(self, stimulus):
        if tuple(stimulus) not in self.hidden_state.keys():
            self._set_hidden_state(stimulus, self._set_a_b())
        return self.observation(stimulus).logpdf

    def act(self, stimulus):
        if tuple(stimulus) not in self.hidden_state.keys():
            self._set_hidden_state(stimulus, self._set_a_b())
        return self.observation(stimulus).rvs()

    def update(self, stimulus, reward, action, done):
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

    def _predict_reward(self, stimulus):
        mix_coef = self.paras['mix_coef']
        b0 = self.paras['b0'] # intercept
        b1 = self.paras['b1'] # slope

        if tuple(stimulus) not in self.hidden_state.keys():
            self._set_hidden_state(stimulus, self._set_a_b())
        a = self._get_hidden_state(stimulus)['a']
        b = self._get_hidden_state(stimulus)['b']

        mu      = beta(a,b).mean()
        entropy = beta(a,b).entropy()

        rhat = b0 + np.dot(stimulus, (mix_coef * mu  + (1 - mix_coef) * entropy)) * b1
        
        return rhat
