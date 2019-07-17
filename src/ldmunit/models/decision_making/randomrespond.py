import numpy as np
from gym import spaces
from scipy import stats

from .base import DADO
from ...capabilities import Interactive, LogProbModel

class RandomRespondModel(DADO, Interactive, LogProbModel):
    """
    Random respond model that predicts random actions for any
    kind of observation.
    """
    name = 'RandomRespondModel'

    def __init__(self, *args, bias, action_bias, **kwargs):
        """
        Parameters
        ----------
        bias : float
            Bias probability. Must be in range [0, 1].

        action_bias : int
            ID of the action. Must be in range [0, n_action)
        """
        assert bias >= 0 and bias <= 1, 'bias must be in range [0, 1]'
        assert np.issubdtype(type(action_bias), np.integer), 'action_bias must be integer'
        paras = dict(bias=bias, action_bias=action_bias)
        super().__init__(paras=paras, **kwargs)
        assert action_bias >= 0 and action_bias < self.n_action, 'action_bias must be in range [0, n_action)'

    def reset(self):
        self.hidden_state = dict()

    def _get_rv(self, stimulus):
        assert self.observation_space.contains(stimulus)

        bias        = self.paras['bias']
        action_bias = self.paras['action_bias']

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
