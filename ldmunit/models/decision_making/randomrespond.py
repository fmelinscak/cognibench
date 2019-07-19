import numpy as np
from gym import spaces
from scipy import stats

from ldmunit.models import DADO
from ldmunit.capabilities import Interactive, LogProbModel


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

        Other Parameters
        ----------------
        **kwargs : any type
            All the mandatory keyword-only arguments required by :class:`ldmunit.models.decision_making.base.DADO` must also be
            provided during initialization.
        """
        assert bias >= 0 and bias <= 1, 'bias must be in range [0, 1]'
        assert np.issubdtype(type(action_bias), np.integer), 'action_bias must be integer'
        paras = dict(bias=bias, action_bias=action_bias)
        super().__init__(paras=paras, **kwargs)
        assert action_bias >= 0 and action_bias < self.n_action, 'action_bias must be in range [0, n_action)'

    def reset(self):
        """
        Override base class reset behaviour by setting the hidden state to default values.
        """
        self.hidden_state = dict()

    def _get_rv(self, stimulus):
        """
        Return a random variable object from the given stimulus.
        """
        assert self.observation_space.contains(stimulus)

        bias = self.paras['bias']
        action_bias = self.paras['action_bias']

        n = self.n_action
        pk = np.full(n, (1 - bias) / (n - 1))
        pk[action_bias] = bias

        xk = np.arange(n)
        rv = stats.rv_discrete(values=(xk, pk))
        rv.random_state = self.seed

        return rv

    def predict(self, stimulus):
        """
        Predict the log-pmf over the discrete action space by using the
        given stimulus as input.

        Parameters
        ----------
        stimulus : int
            A stimulus from the observation space for this model.

        Returns
        -------
        method
            :py:meth:`scipy.stats.rv_discrete.logpmf` method over the discrete action space.
        """
        return self._get_rv(stimulus).logpmf

    def act(self, stimulus):
        """
        Return an action for the given stimulus.

        Parameters
        ----------
        stimulus : int
            A stimulus from the observation space for this model.

        Returns
        -------
        int
            An action from the action space.
        """
        return self._get_rv(stimulus).rvs()

    def update(self, stimulus, reward, action, done):
        """
        Doesn't do anything. Stimulus and action must be from their respective
        spaces.
        """
        assert self.action_space.contains(action)
        assert self.observation_space.contains(stimulus)
