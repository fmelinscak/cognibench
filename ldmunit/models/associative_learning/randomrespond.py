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


class RandomRespondAgent(
    LDMAgent, ProducesPolicy, ContinuousAction, MultiBinaryObservation
):
    """
    Random respond model that predicts random actions for any
    kind of observation.
    """

    name = "RandomRespond"

    @overrides
    def __init__(self, *args, n_obs, **kwargs):
        """
        Parameters
        ----------
        n_obs : int
            Size of the observation space

        paras_dict : dict (optional)
            mu : float
                Mean of the normal random variables used to predict actions
                and rewards.

            sigma : float
                Standard deviation of the normal random variables used to predict
                actions and rewards. Must be nonnegative.
        """
        self.set_action_space(ContinuousSpace())
        self.set_observation_space(n_obs)
        super().__init__(*args, **kwargs)

    def reset(self):
        """
        Reset the hidden state to its default value.
        """
        self.hidden_state = dict()

    def eval_policy(self, stimulus):
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
        assert self.get_observation_space().contains(stimulus)

        mu_pred = self.paras["mu"]
        sd_pred = self.paras["sigma"]

        rv = stats.norm(loc=mu_pred, scale=sd_pred)
        rv.random_state = self.seed

        return rv

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
        return self.eval_policy(stimulus).rvs()

    def update(self, stimulus, reward, action, done=False):
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
        assert self.get_action_space().contains(action)
        assert self.get_observation_space().contains(stimulus)


class RandomRespondModel(PolicyModel, ContinuousAction, MultiBinaryObservation):
    """
    Random respond model implementation.
    """

    name = "Random respond"

    @overrides
    def __init__(self, *args, n_obs, seed=None, **kwargs):
        self.set_action_space(ContinuousSpace())
        self.set_observation_space(n_obs)
        agent = RandomRespondAgent(n_obs=n_obs, seed=seed)

        def initializer(seed):
            return {
                "mu": stats.norm.rvs(scale=0, random_state=seed),
                "sigma": stats.expon.rvs(scale=2, random_state=seed),
            }

        super().__init__(
            *args, agent=agent, param_initializer=initializer, seed=seed, **kwargs
        )
