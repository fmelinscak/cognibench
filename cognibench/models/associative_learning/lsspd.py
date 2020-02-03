import numpy as np
import gym
from gym import spaces
from scipy import stats
from cognibench.models import CNBAgent
from cognibench.models.policy_model import PolicyModel
from cognibench.capabilities import (
    ProducesPolicy,
    ContinuousAction,
    MultiBinaryObservation,
)
from cognibench.continuous import ContinuousSpace
from cognibench.utils import is_arraylike
from overrides import overrides


class LSSPDAgent(CNBAgent, ProducesPolicy, ContinuousAction, MultiBinaryObservation):
    def __init__(self, *args, n_obs, **kwargs):
        """
        Parameters
        ----------
        n_obs : int
            Size of the observation space

        paras_dict : dict (optional)
            w : array-like
                Initial value of weight vector w. It must have the same length as the dimension of the observation space.

            alpha : array-like
                Initial value of associability vector alpha. It must have the same length as the dimension of the observation space.

            b0 : float
                Intercept used when computing the mean of normal distribution from reward.

            b1 : array-like
                Slope used when computing the mean of the normal distribution from reward. It must have the same length as the dimension of the observation space.

            sigma : float
                Standard deviation of the normal distribution used to generate observations.
                Must be nonnegative.

            mix_coef : float
                Mixing coefficient used in the convex combination of weight and associability vectors.
                Must be in [0, 1] range.

            eta : float
                Learning rate for alpha updates. Must be nonnegative.

            kappa : float
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

        eta = self.get_paras()[
            "eta"
        ]  # Proportion of pred. error. in the updated associability value
        kappa = self.get_paras()[
            "kappa"
        ]  # Fixed learning rate for the cue weight update

        w_curr = self.get_hidden_state()["w"]
        alpha = self.get_hidden_state()["alpha"]

        rhat = self._predict_reward(stimulus)

        if not done:
            delta = reward - rhat

            w_curr += kappa * delta * alpha * stimulus  # alpha, stimulus size: (n_obs,)

            alpha += stimulus * (eta * abs(delta) - eta * alpha)
            alpha = np.minimum(alpha, 1)

            self.get_hidden_state()["w"] = w_curr
            self.get_hidden_state()["alpha"] = alpha

        return w_curr, alpha

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
        assert self.get_observation_space().contains(stimulus)

        b0 = self.get_paras()["b0"]
        b1 = self.get_paras()["b1"]
        sd_pred = self.get_paras()["sigma"]
        mix_coef = self.get_paras()["mix_coef"]

        w_curr = self.get_hidden_state()["w"]
        alpha = self.get_hidden_state()["alpha"]

        # Predict response
        mu_pred = b0 + np.dot(
            b1, stimulus * (mix_coef * w_curr + (1 - mix_coef) * alpha)
        )

        rv = stats.norm(loc=mu_pred, scale=sd_pred)
        rv.random_state = self.get_seed()

        return rv

    @overrides
    def reset(self):
        """
        Reset the hidden state to its default value.
        """
        w = self.get_paras()["w"]
        alpha = self.get_paras()["alpha"]
        self.set_hidden_state({"w": w, "alpha": alpha})


class LSSPDModel(PolicyModel, ContinuousAction, MultiBinaryObservation):
    """
    LSSPD (Rescorla Wagner Pearce Hall, RWPH) model implementation.
    """

    # RWPH
    name = "LSSPD"

    @overrides
    def __init__(self, *args, n_obs, seed=None, **kwargs):
        self.set_action_space(ContinuousSpace())
        self.set_observation_space(n_obs)
        agent = LSSPDAgent(n_obs=n_obs, seed=seed)

        def initializer(seed):
            return {
                "w": stats.norm.rvs(size=n_obs, random_state=seed),
                "alpha": stats.norm.rvs(size=n_obs, random_state=seed),
                "sigma": stats.expon.rvs(random_state=seed),
                "b0": stats.norm.rvs(random_state=seed),
                "b1": stats.norm.rvs(size=n_obs, random_state=seed),
                "mix_coef": stats.uniform.rvs(random_state=seed),
                "eta": 1e-3,
                "kappa": 1e-3,
            }

        self.param_bounds = {
            "w": [None] * 2 * n_obs,
            "alpha": [0, 1] * n_obs,
            "sigma": (0, 1),
            "b0": (None, None),
            "b1": [None] * 2 * n_obs,
            "mix_coef": (0, 1),
            "eta": (0, 1),
            "kappa": (0, 1),
        }

        super().__init__(
            *args, agent=agent, param_initializer=initializer, seed=seed, **kwargs
        )
