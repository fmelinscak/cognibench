import sciunit
import numpy as np
from gym import spaces
from scipy import stats

from .base import DADO

class RandomRespondModel(DADO):
    """Random respond for discrete decision marking."""

    def __init__(self, n_action=None, n_obs=None, paras=None, hidden_state=None, seed=None, name=None, **params):
        return super().__init__(n_action=n_action, n_obs=n_obs, paras=paras, hidden_state=hidden_state, seed=seed, name=name, **params)
        
    def reset(self):
        if not (isinstance(self.n_action, int) and isinstance(self.n_obs, int)):
            raise TypeError("action_space and observation_space must be set.")

        self.hidden_state = None

    def _get_rv(self, stimulus, paras=None):
        assert self.observation_space.contains(stimulus)
        if not paras:
            paras = self.paras
        bias        = paras['bias']
        action_bias = paras['action_bias']

        n = self.n_action
        pk = np.full(n, (1 - bias) / (n - 1))
        pk[action_bias] = bias
        
        xk = np.arange(n)
        rv = stats.rv_discrete(values=(xk, pk))
        if self.seed:
            rv.random_state = self.seed

        return rv

    def predict(self, stimulus):
        return self._get_rv(stimulus).logpmf

    def act(self, stimulus):
        return self._get_rv(stimulus).rvs()

    def update(self, stimulus, reward, action, done):
        assert self.action_space.contains(action)
        assert self.observation_space.contains(stimulus)
        pass



