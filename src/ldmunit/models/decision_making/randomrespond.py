import numpy as np
from gym import spaces
from scipy import stats

from .base import DADO
from ...capabilities import Interactive, LogProbModel

class RandomRespondModel(DADO, Interactive, LogProbModel):
    """Random respond for discrete decision marking."""
    name = 'RandomRespondModel'

    def __init__(self, *args, bias, action_bias, **kwargs):
        paras = dict(bias=bias, action_bias=action_bias)
        super().__init__(paras=paras, **kwargs)

    def reset(self):
        self.hidden_state = dict()

    def _get_rv(self, stimulus):
        assert self.observation_space.contains(stimulus)

        bias        = self.paras['bias']
        action_bias = self.paras['action_bias']
        action_bias = int(action_bias) if isinstance(action_bias, float) else action_bias

        n = self.n_action
        pk = np.full(n, (1 - bias) / (n - 1))
        pk[action_bias] = bias
        
        xk = np.arange(n)
        rv = stats.rv_discrete(values=(xk, pk))
        rv.random_state = self.seed

        return rv

    def predict(self, stimulus):
        return self._get_rv(stimulus).logpmf

    def act(self, stimulus):
        return self._get_rv(stimulus).rvs()

    def update(self, stimulus, reward, action, done):
        assert self.action_space.contains(action)
        assert self.observation_space.contains(stimulus)
