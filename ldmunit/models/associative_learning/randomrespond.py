import numpy as np
import gym
from gym import spaces
from scipy import stats
from ldmunit.models import CAMO, ParametricModelMixin
from ldmunit.capabilities import Interactive, PredictsLogpdf
from overrides import overrides


class RandomRespondModel(CAMO, Interactive, PredictsLogpdf, ParametricModelMixin):
    """
    Random respond model that predicts random actions for any
    kind of observation.
    """

    name = "RandomRespond"

    @overrides
    def __init__(self, *args, mu, sigma, **kwargs):
        """
        Parameters
        ----------
        mu : float
            Mean of the normal random variables used to predict actions
            and rewards.

        sigma : float
            Standard deviation of the normal random variables used to predict
            actions and rewards. Must be nonnegative.

        Other Parameters
        ----------------
        **kwargs : any type
            All the mandatory keyword-only arguments required by :class:`ldmunit.models.associative_learning.base.CAMO` must also be
            provided during initialization.
        """
        assert sigma >= 0, "sigma must be nonnegative"
        paras = dict(mu=mu, sigma=sigma)
        super().__init__(paras=paras, **kwargs)

    def reset(self):
        """
        Reset the hidden state to its default value.
        """
        self.hidden_state = dict()

    def observation(self, stimulus):
        """
        Get the random variable for the given stimulus.

        Parameters
        ----------
        stimulus : :class:`np.ndarray`
            Single stimulus from the observation space.

        Returns
        -------
        :class:`scipy.stats.rv_continuous`
            Normal random variable with mean equal to reward and
            standard deviation equal to sigma model parameter.
        """
        assert self.observation_space.contains(stimulus)

        mu_pred = self.paras["mu"]
        sd_pred = self.paras["sigma"]

        rv = stats.norm(loc=mu_pred, scale=sd_pred)
        rv.random_state = self.seed

        return rv

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

    def update(self, stimulus, reward, action, done):
        """
        Update the hidden state of the model based on input stimulus, action performed
        by the model and reward.

        Parameters
        ----------
        stimulus : array-like
            A stimulus from the multi-binary observation space for this model. For
            example, `[0, 1, 1]`.

        action : float
            The action performed by the model.
        """
        assert self.action_space.contains(action)
        assert self.observation_space.contains(stimulus)
