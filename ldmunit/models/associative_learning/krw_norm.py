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


class KrwNormAgent(LDMAgent, ProducesPolicy, ContinuousAction, MultiBinaryObservation):
    def __init__(self, *args, n_obs, **kwargs):
        """
        Parameters
        ----------
        n_obs : int
            Size of the observation space

        paras_dict : dict (optional)
            Dictionary containing the agent parameters, as explained below:

            w : array-like
                Initial value of weight vector w. It must have the same length as the dimension of the observation space.

            sigma : float
                Standard deviation of the normal distribution used to generate observations.
                Must be nonnegative.

            b0 : float
                Intercept used when computing the mean of normal distribution from reward.

            b1 : array-like
                Slope used when computing the mean of the normal distribution from reward.
                It must have the same length as the dimension of the observation space.

            sigmaWInit : float
                Diagonal elements of the covariance matrix C is set to sigmaWInit.

            tauSq : float
                Diagonal elements of the transition noise variance matrix Q is set to tauSq.
                Must be nonnegative.

            sigmaRSq : float
                Additive factor used in the denominator when computing the Kalman gain K.
                Must be nonnegative.
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

        tauSq = self.get_paras()["tauSq"]  # State diffusion variance
        Q = tauSq * np.identity(
            self.n_obs()
        )  # Transition noise variance (transformed to positive reals); constant over time
        sigmaRSq = self.get_paras()["sigmaRSq"]

        w_curr = self.get_hidden_state()["w"]
        C_curr = self.get_hidden_state()["C"]

        rhat = self._predict_reward(stimulus)

        if not done:
            # Kalman prediction step
            w_pred = w_curr  # No mean-shift for the weight distribution evolution (only stochastic evolution)
            C_pred = C_curr + Q  # Update covariance

            # get pred_error
            delta = reward - rhat

            # Kalman update step
            K = C_pred.dot(stimulus) / (
                stimulus.dot(C_pred.dot(stimulus)) + sigmaRSq
            )  # (n_obs(),)
            w_updt = w_pred + K * delta  # Mean updated with prediction error
            C_updt = C_pred - np.dot(K[:, None], np.dot(stimulus[None, :], C_pred))

            self.get_hidden_state()["w"] = w_updt
            self.get_hidden_state()["C"] = C_updt

        return w_updt, C_updt

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
        C = self.get_paras()["sigmaWInit"] * np.identity(self.n_obs())

        self.set_hidden_state({"w": np.full(self.n_obs(), w), "C": C})


class KrwNormModel(PolicyModel, ContinuousAction, MultiBinaryObservation):
    """
    Kalman Rescorla-Wagner model implementation.
    """

    name = "KrwNorm"

    @overrides
    def __init__(self, *args, n_obs, seed=None, **kwargs):
        self.set_action_space(ContinuousSpace())
        self.set_observation_space(n_obs)
        agent = KrwNormAgent(n_obs=n_obs, seed=seed)

        def initializer(seed):
            return {
                "tauSq": stats.expon.rvs(random_state=seed),
                "sigmaRSq": stats.expon.rvs(random_state=seed),
                "w": stats.norm.rvs(size=n_obs, random_state=seed),
                "sigma": stats.expon.rvs(random_state=seed),
                "b0": stats.norm.rvs(random_state=seed),
                "b1": stats.norm.rvs(size=n_obs, random_state=seed),
                "sigmaWInit": np.ones(n_obs),
            }

        super().__init__(
            *args, agent=agent, param_initializer=initializer, seed=seed, **kwargs
        )
