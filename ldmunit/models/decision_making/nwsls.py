import numpy as np
from gym import spaces
from scipy import stats

from ldmunit.models import DADO, ParametricModelMixin
from ldmunit.capabilities import Interactive, PredictsLogpdf


class NWSLSModel(DADO, Interactive, PredictsLogpdf, ParametricModelMixin):
    """
    Noisy-win-stay-lose-shift model implementation.
    """

    name = "NWSLSModel"

    def __init__(self, *args, epsilon, **kwargs):
        """
        Parameters
        ----------
        epsilon : int
            Number of loose actions. Must be nonnegative and less than or equal
            to the dimension of the action space.

        Other Parameters
        ----------------
        **kwargs : any type
            All the mandatory keyword-only arguments required by :class:`ldmunit.models.decision_making.base.DADO` must also be
            provided during initialization.
        """
        paras = dict(epsilon=epsilon)
        super().__init__(paras=paras, **kwargs)
        assert (
            epsilon >= 0 and epsilon <= self.n_action
        ), "epsilon must be in range [0, n_action]"

    def reset(self):
        """
        Override base class reset behaviour by setting the hidden state to default
        values for NWSLS model.
        """
        self.hidden_state = dict(win=True, action=self.rng.randint(0, self.n_action))

    def _get_rv(self, stimulus):
        """
        Return a random variable object from the given stimulus.
        """
        assert self.observation_space.contains(stimulus)

        epsilon = self.paras["epsilon"]
        n = self.n_action

        if self.hidden_state["win"]:
            prob_action = 1 - epsilon / n
        else:
            prob_action = epsilon / n

        pk = np.full(n, (1 - prob_action) / (n - 1))
        pk[self.hidden_state["action"]] = prob_action

        xk = np.arange(n)
        rv = stats.rv_discrete(name=None, values=(xk, pk))
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
        Update the hidden state of the model based on input stimulus, action performed
        by the model and reward.

        Parameters
        ----------
        stimulus : int
            A stimulus from the observation space for this model.

        reward : int
            The reward for the action.

        action : int
            Action performed by the model.
        """
        assert self.action_space.contains(action)
        assert self.observation_space.contains(stimulus)

        self.hidden_state["win"] = reward == 1
        self.hidden_state["action"] = action
