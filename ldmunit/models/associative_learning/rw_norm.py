import numpy as np
import gym
from gym import spaces
from scipy import stats
from ldmunit.models import LDMAgent
from ldmunit.models.policy_model import PolicyModel
from ldmunit.capabilities import (
    ProducesPolicy,
    ContinuousAction,
    MultiBinaryObservation,
)
from ldmunit.continuous import ContinuousSpace
from ldmunit.utils import is_arraylike
from overrides import overrides


class RwNormAgent(LDMAgent, ProducesPolicy, ContinuousAction, MultiBinaryObservation):
    def __init__(self, *args, n_obs, **kwargs):
        """
        Parameters
        ----------
        n_obs : int
            Size of the observation space

        paras_dict : dict (optional)
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
        """

        self.set_action_space(ContinuousSpace())
        self.set_observation_space(n_obs)
        # TODO: get params here
        super().__init__(*args, **kwargs)

    @overrides
    def act(self, *args, **kwargs):
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
        return self.eval_policy(*args, **kwargs).rvs()

    def _predict_reward(self, stimulus):
        assert self.get_observation_space().contains(stimulus)
        w_curr = self.get_hidden_state()["w"]
        rhat = np.dot(stimulus, w_curr.T)
        return rhat

    @overrides
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
        assert self.get_action_space().contains(action)
        assert self.get_observation_space().contains(stimulus)

        eta = self.get_paras()["eta"]
        w_curr = self.get_hidden_state()["w"]

        rhat = self._predict_reward(stimulus)

        if not done:
            delta = reward - rhat
            w_curr += eta * delta * stimulus
            self.get_hidden_state()["w"] = w_curr

        return w_curr

    @overrides
    def eval_policy(self, stimulus):
        """
        Get the action random variable for the given stimulus.

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
        assert self.get_hidden_state(), "hidden state must be set"
        assert self.get_observation_space().contains(stimulus)

        b0 = self.get_paras()["b0"]  # intercept
        b1 = self.get_paras()["b1"]  # slope
        sd_pred = self.get_paras()["sigma"]

        w_curr = self.get_hidden_state()["w"]

        # Predict response
        mu_pred = b0 + np.dot(b1, stimulus * w_curr)

        rv = stats.norm(loc=mu_pred, scale=sd_pred)
        rv.random_state = self.get_seed()

        return rv

    @overrides
    def reset(self):
        """
        Reset the hidden state to its default value.
        """
        w = self.get_paras()["w"]
        self.set_hidden_state({"w": w})


class RwNormModel(PolicyModel, ContinuousAction, MultiBinaryObservation):
    """
    Rescorla-Wagner model implementation.
    """

    name = "RwNorm"

    @overrides
    def __init__(self, *args, n_obs, seed=None, **kwargs):
        self.set_action_space(ContinuousSpace())
        self.set_observation_space(n_obs)
        agent = RwNormAgent(n_obs=n_obs, seed=seed)

        def initializer(seed):
            return {
                "w": stats.norm.rvs(size=n_obs, random_state=seed),
                "sigma": stats.expon.rvs(random_state=seed),
                "b0": stats.norm.rvs(random_state=seed),
                "b1": stats.norm.rvs(size=n_obs, random_state=seed),
                "eta": np.ones(n_obs) * 1e-3,
            }

        super().__init__(
            *args, agent=agent, param_initializer=initializer, seed=seed, **kwargs
        )
