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
from collections.abc import MutableMapping
from cognibench.continuous import ContinuousSpace
from cognibench.utils import is_arraylike
from overrides import overrides


class BetaBinomialAgent(
    CNBAgent, ProducesPolicy, ContinuousAction, MultiBinaryObservation
):
    name = "BetaBinomial"

    def __init__(self, *args, n_obs, **kwargs):
        """
        Parameters
        ----------
        n_obs : int
            Size of the observation space

        paras_dict : dict (optional)
            a : float
                Initial value of occurence count variable `a`. Must be positive.

            b : float
                Initial value of non-occurence count variable `b`. Must be positive.

            sigma : float
                Standard deviation of the normal distribution used to generate observations.
                Must be nonnegative.

            mix_coef : float
                Mixing coefficient used in the convex combination. Must be in [0, 1] range.

            intercept : float
                Intercept used when computing the reward.

            slope : array-like or float
                Slope used when computing the reward. If a single scalar is given, each element of the slope vector
                is equal to that value.
        """
        self.set_action_space(ContinuousSpace())
        self.set_observation_space(n_obs)
        super().__init__(*args, **kwargs)

    def _get_default_a_b(self):
        """
        Get default occurence and non-occurence counts.
        """
        a = self.get_paras()["a"]
        b = self.get_paras()["b"]
        out = {
            "a": a * np.ones(self.n_obs(), dtype=np.float64),
            "b": b * np.ones(self.n_obs(), dtype=np.float64),
        }
        return out

    def reset(self):
        """
        Reset the hidden state to its default value.
        """
        self.set_hidden_state(_DictWithBinarySequenceKeys())

    def eval_policy(self, stimulus):
        """
        Get the reward random variable for the given stimulus.

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

        sd_pred = self.get_paras()["sigma"]

        mu_pred = self._predict_reward(stimulus)

        rv = stats.norm(loc=mu_pred, scale=sd_pred)
        rv.random_state = self.get_seed()

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

        reward : float
            The reward for the action.

        done : bool
            If True, do not update the hidden state.
        """
        assert self.get_observation_space().contains(stimulus)
        # get model's state
        if tuple(stimulus) not in self.get_hidden_state().keys():
            self.get_hidden_state()[stimulus] = self._get_default_a_b()
        a = self.get_hidden_state()[stimulus]["a"]
        b = self.get_hidden_state()[stimulus]["b"]

        if not done:
            a += stimulus * reward
            b += stimulus * (1 - reward)
            self.get_hidden_state()[stimulus]["a"] = a
            self.get_hidden_state()[stimulus]["b"] = b

        return a, b

    def _predict_reward(self, stimulus):
        """
        Predict the reward from the given stimulus using beta-binomial model
        equations.
        """
        mix_coef = self.get_paras()["mix_coef"]
        intercept = self.get_paras()["intercept"]
        slope = self.get_paras()["slope"]

        if tuple(stimulus) not in self.get_hidden_state().keys():
            self.get_hidden_state()[stimulus] = self._get_default_a_b()
        a = self.get_hidden_state()[stimulus]["a"]
        b = self.get_hidden_state()[stimulus]["b"]

        mu = stats.beta(a, b).mean()
        entropy = stats.beta(a, b).entropy()

        rhat = intercept + np.dot(
            slope, (mix_coef * mu + (1 - mix_coef) * entropy) * stimulus
        )

        return rhat


class BetaBinomialModel(PolicyModel, ContinuousAction, MultiBinaryObservation):
    """
    Beta-binomial model implementation.
    """

    name = "Beta Binomial"

    @overrides
    def __init__(self, *args, n_obs, seed=None, **kwargs):
        self.set_action_space(ContinuousSpace())
        self.set_observation_space(n_obs)
        agent = BetaBinomialAgent(n_obs=n_obs, seed=seed)

        def initializer(seed):
            return {
                "a": stats.expon.rvs(scale=5, random_state=seed),
                "b": stats.expon.rvs(scale=5, random_state=seed),
                "sigma": stats.expon.rvs(random_state=seed),
                "mix_coef": stats.uniform.rvs(random_state=seed),
                "intercept": stats.norm.rvs(random_state=seed),
                "slope": stats.norm.rvs(size=n_obs, random_state=seed),
            }

        self.param_bounds = {
            "a": (1, None),
            "b": (1, None),
            "sigma": (1e-6, None),
            "mix_coef": (0, 1),
            "intercept": (None, None),
            "slope": [None] * 2 * n_obs,
        }

        super().__init__(
            *args, agent=agent, param_initializer=initializer, seed=seed, **kwargs
        )


class _DictWithBinarySequenceKeys(MutableMapping):
    """
    Mapping where keys are binary sequences such as [0, 1, 1], [1, 0, 1], etc.
    """

    def __init__(self, *args, **kwargs):
        self._storage = dict()
        self.update(dict(*args, **kwargs))

    def __getitem__(self, key):
        return self._storage[tuple(key)]

    def __setitem__(self, key, val):
        self._storage[tuple(key)] = val

    def __delitem__(self, key):
        del self._storage[tuple(key)]

    def __iter__(self):
        return iter(self._storage)

    def __len__(self):
        return len(self._storage)
