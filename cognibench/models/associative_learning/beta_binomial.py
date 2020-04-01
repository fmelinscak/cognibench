import numpy as np
import gym
from collections import defaultdict
from gym import spaces
from scipy import stats
from cognibench.distr import NormalRV
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

    def __init__(self, *args, n_obs, distinct_stimuli, **kwargs):
        """
        Parameters
        ----------
        n_obs : int
            Size of the observation space

        distinct_stimuli : array-like
            List of distinct stimuli that can be passed to this agent. Each element
            should be an array-like object that contain only 0s and 1s. Further, all
            the element must have the same size.

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
        self.n_cues = len(distinct_stimuli)
        self.cue_to_idx = _CueToIdxMap()
        for i, stimulus in enumerate(distinct_stimuli):
            self.cue_to_idx[stimulus] = i
        self.set_action_space(ContinuousSpace())
        self.set_observation_space(n_obs)
        super().__init__(*args, **kwargs)

    @overrides
    def set_paras(self, paras):
        if paras is None:
            self._paras = paras
            return
        own_paras = {k: v for k, v in paras.items()}
        self.a_init = own_paras["a"]
        self.b_init = own_paras["b"]
        del own_paras["a"]
        del own_paras["b"]
        self._paras = own_paras
        self.reset()

    def reset(self):
        self.set_hidden_state(
            {
                "a": self.a_init * np.ones(self.n_cues),
                "b": self.a_init * np.ones(self.n_cues),
            }
        )

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

        rv = NormalRV(loc=mu_pred, scale=sd_pred)
        rv.random_state = self.rng

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
        hidden = self.get_hidden_state()

        stimulus_idx = self.cue_to_idx[stimulus]
        if not done:
            hidden["a"][stimulus_idx] += reward
            hidden["b"][stimulus_idx] += 1 - reward

    def _predict_reward(self, stimulus):
        """
        Predict the reward from the given stimulus using beta-binomial model
        equations.
        """
        mix_coef = self.get_paras()["mix_coef"]
        intercept = self.get_paras()["intercept"]
        slope = self.get_paras()["slope"]

        stimulus_idx = self.cue_to_idx[stimulus]
        hidden = self.get_hidden_state()

        a_i = hidden["a"][stimulus_idx]
        b_i = hidden["b"][stimulus_idx]
        slope_i = slope[stimulus_idx]

        mu_i = stats.beta.mean(a_i, b_i)
        entropy_i = stats.beta.entropy(a_i, b_i)

        rhat = intercept + slope_i * (mix_coef * mu_i + (1 - mix_coef) * entropy_i)
        # rhat = intercept + np.dot(
        #    slope,
        #    (mix_coef * mu + (1 - mix_coef) * entropy) * stimulus
        # )

        return rhat


class BetaBinomialModel(PolicyModel, ContinuousAction, MultiBinaryObservation):
    """
    Beta-binomial model implementation.
    """

    name = "Beta Binomial"

    @overrides
    def __init__(self, *args, n_obs, distinct_stimuli, seed=None, **kwargs):
        self.set_action_space(ContinuousSpace())
        self.set_observation_space(n_obs)
        agent = BetaBinomialAgent(
            n_obs=n_obs, distinct_stimuli=distinct_stimuli, seed=seed
        )

        n_cues = len(distinct_stimuli)

        def initializer(seed):
            return {
                "a": 1,
                "b": 1,
                "sigma": stats.expon.rvs(random_state=seed),
                "mix_coef": stats.uniform.rvs(random_state=seed),
                "intercept": stats.norm.rvs(random_state=seed),
                "slope": stats.norm.rvs(size=n_cues, random_state=seed),
            }

        self.param_bounds = {
            "sigma": (1e-6, None),
            "mix_coef": (0, 1),
            "intercept": (None, None),
            "slope": [None] * 2 * n_cues,
        }

        super().__init__(
            *args, agent=agent, param_initializer=initializer, seed=seed, **kwargs
        )


class _CueToIdxMap(MutableMapping):
    """
    Mapping where keys are binary sequences such as [0, 1, 1], [1, 0, 1], etc.
    """

    def __init__(self):
        self._storage = dict()

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
