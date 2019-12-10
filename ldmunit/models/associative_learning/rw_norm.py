import numpy as np
import gym
from gym import spaces
from scipy import stats
from ldmunit.models import CAMO
from ldmunit.models.mixins import (
    ParametricModelMixin,
    ReinforcementLearningFittingMixin,
)
from ldmunit.capabilities import Interactive, PredictsLogpdf
from ldmunit.utils import is_arraylike
from overrides import overrides


class RwNormModel(
    ParametricModelMixin,
    ReinforcementLearningFittingMixin,
    CAMO,
    Interactive,
    PredictsLogpdf,
):
    """
    Rescorla-Wagner model implementation.
    """

    name = "RwNorm"

    @overrides
    def __init__(self, *args, w, b0, b1, sigma, eta, **kwargs):
        """
        Parameters
        ----------
        w : float or array-like
            Initial value of weight vector w. If float, then all elements of the
            weight vector is set to this value. If array-like, must have the same
            length as the dimension of the observation space.

        sigma : float
            Standard deviation of the normal distribution used to generate observations.
            Must be nonnegative.

        b0 : float
            Intercept used when computing the mean of normal distribution from reward.

        b1 : array-like or float
            Slope used when computing the mean of the normal distribution from reward.
            If a scalar is given, all elements of the slope vector is equal to that value.

        eta : float
            Learning rate for w updates. Must be nonnegative.

        Other Parameters
        ----------------
        **kwargs : any type
            All the mandatory keyword-only arguments required by :class:`ldmunit.models.associative_learning.base.CAMO` must also be
            provided during initialization.
        """
        assert sigma >= 0, "sigma must be nonnegative"
        assert eta >= 0, "eta must be nonnegative"
        paras = {"w": w, "b0": b0, "b1": b1, "sigma": sigma, "eta": eta}
        super().__init__(paras=paras, **kwargs)
        if is_arraylike(w):
            assert (
                len(w) == self.n_obs()
            ), "w must have the same length as the dimension of the observation space"
        if not is_arraylike(b1):
            self.paras["b1"] = np.full(self.n_obs(), b1)

    def reset(self):
        w = self.paras["w"] if "w" in self.paras else 0
        if is_arraylike(w):
            w = np.array(w, dtype=np.float64)
        else:
            w = np.full(self.n_obs(), w, dtype=np.float64)

        self.hidden_state = {"w": w}

    def predict(self, stimulus):
        """
        Predict the log-pdf over the continuous action space by using the
        given stimulus as input.

        Parameters
        ----------
        stimulus : array-like
            A stimulus from the multi-binary observation space for this model. For
            example, `[0, 1, 1]`.

        Returns
        -------
        method
            :py:meth:`scipy.stats.rv_continuous.logpdf` method over the continuous action space.
        """
        return self.observation(stimulus).logpdf

    def act(self, stimulus):
        """
        Return an action for the given stimulus.

        Parameters
        ----------
        stimulus : array-like
            A stimulus from the multi-binary observation space for this model. For
            example, `[0, 1, 1]`.

        Returns
        -------
        float
            An action from the continuous action space.
        """
        return self.observation(stimulus).rvs()

    def _predict_reward(self, stimulus):
        assert self.observation_space().contains(stimulus)
        w_curr = self.hidden_state["w"]
        rhat = np.dot(stimulus, w_curr.T)
        return rhat

    def observation(self, stimulus):
        """
        Get the reward random variable for the given stimulus.

        Parameters
        ----------
        stimulus : array-like
            Single stimulus from the observation space.

        Returns
        -------
        :class:`scipy.stats.rv_continuous`
            Normal random variable with mean equal to linearly transformed
            reward using b0 and b1 parameters, and standard deviation equal
            to sigma model parameter.
        """
        assert self.hidden_state, "hidden state must be set"
        assert self.observation_space().contains(stimulus)

        b0 = self.paras["b0"]  # intercept
        b1 = self.paras["b1"]  # slope
        sd_pred = self.paras["sigma"]

        w_curr = self.hidden_state["w"]

        # Predict response
        mu_pred = b0 + np.dot(b1, stimulus * w_curr)

        rv = stats.norm(loc=mu_pred, scale=sd_pred)
        rv.random_state = self.seed

        return rv

    def update(self, stimulus, reward, action, done=False):
        """
        Update the hidden state of the model based on input stimulus, action performed
        by the model and reward.

        Parameters
        ----------
        stimulus : array-like
            A stimulus from the multi-binary observation space for this model. For
            example, `[0, 1, 1]`.

        reward : float
            The reward for the action.

        action : float
            Action performed by the model.

        done : bool
            If True, do not update the hidden state.
        """
        assert self.action_space().contains(action)
        assert self.observation_space().contains(stimulus)

        eta = self.paras["eta"]
        w_curr = self.hidden_state["w"]

        rhat = self._predict_reward(stimulus)

        if not done:
            delta = reward - rhat
            w_curr += eta * delta * stimulus
            self.hidden_state["w"] = w_curr

        return w_curr
