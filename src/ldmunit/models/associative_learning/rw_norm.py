import numpy as np
import gym
from gym import spaces
from scipy import stats
from .base import CAMO
from ...capabilities import Interactive, LogProbModel

class RwNormModel(CAMO, Interactive, LogProbModel):
    name = 'RwNorm'

    def __init__(self, *args, w0, b0, b1, sigma, alpha, **kwargs):
        paras = {
            'w0' : w0,
            'b0' : b0,
            'b1' : b1,
            'sigma' : sigma,
            'alpha' : alpha
        }
        super().__init__(paras=paras, **kwargs)

    def reset(self):
        w0 = self.paras['w0'] if 'w0' in self.paras else 0
        try:
            it = iter(w0)
            w0 = np.array(w0, dtype=np.float64)
        except TypeError:
            w0 = np.full(self.n_obs, w0, dtype=np.float64)

        self.hidden_state = {'w': w0}

    def predict(self, stimulus):
        return self.observation(stimulus).logpdf

    def act(self, stimulus):
        return self.observation(stimulus).rvs()

    def _predict_reward(self, stimulus):
        assert self.observation_space.contains(stimulus)
        w_curr = self.hidden_state['w']
        rhat = np.dot(stimulus, w_curr.T)
        return rhat

    def observation(self, stimulus):
        assert self.hidden_state, "hidden state must be set"
        assert self.observation_space.contains(stimulus)

        b0 = self.paras['b0'] # intercept
        b1 = self.paras['b1'] # slope
        sd_pred = self.paras['sigma']
        
        w_curr = self.hidden_state['w']

        rhat = self._predict_reward(stimulus)

        # Predict response
        mu_pred = b0 + b1 * rhat

        rv = stats.norm(loc=mu_pred, scale=sd_pred)
        rv.random_state = self.seed

        return rv

    def update(self, stimulus, reward, action, done):
        assert self.action_space.contains(action)
        assert self.observation_space.contains(stimulus)

        alpha  = self.paras['alpha']
        w_curr = self.hidden_state['w']

        rhat = self._predict_reward(stimulus)

        if not done:
            delta = reward - rhat
            w_curr += alpha * delta * stimulus
            self.hidden_state['w'] = w_curr

        return w_curr
