import sciunit
import numpy as np
import gym
from gym import spaces
from scipy import stats
from .base import CAMO

class RwNormModel(CAMO):

    def __init__(self, n_obs=None, paras=None, hidden_state=None, name=None, **params):
        return super().__init__(n_obs=n_obs, paras=paras, hidden_state=hidden_state, name=name, **params)

    def reset(self):
        w0 = self.paras['w0'] if 'w0' in self.paras else 0
        w0 = np.array(w0, dtype=np.float64) if isinstance(w0, list) else np.full(self.n_obs, w0, dtype=np.float64)

        hidden_state = {'w': w0}
        self.hidden_state = hidden_state

    def _get_default_paras(self):
        return {'w0': 0.1, 'alpha': 0.5, 'sigma': 0.5, 'b0': 0.5, 'b1': 0.5}

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
        assert isinstance(self.observation_space, spaces.MultiBinary), "observation space must be set first"
        assert self.hidden_state, "hidden state must be set"
        assert self.observation_space.contains(stimulus)

        b0 = self.paras['b0'] # intercept
        b1 = self.paras['b1'] # slope
        sd_pred = self.paras['sigma']
        
        w_curr = self.hidden_state['w']

        rhat = self._predict_reward(stimulus)

        # Predict response
        mu_pred = b0 + b1 * rhat
        return stats.norm(loc=mu_pred, scale=sd_pred)

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

