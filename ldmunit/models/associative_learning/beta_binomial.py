import numpy as np
import gym
from gym import spaces
from scipy import stats
from scipy.stats import beta
from ldmunit.models import CAMO
from ldmunit.capabilities import Interactive, LogProbModel
from collections.abc import MutableMapping


class DictWithBinarySequenceKeys(MutableMapping):
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


class BetaBinomialModel(CAMO, Interactive, LogProbModel):
    """
    Interactive beta-binomial model implementation.

    Occurence and non-occurence counts are stored in variables `a` and `b`, respectively.

    Example
    -------
    >>> # Reward is calculated as
    >>> mu = mean(Beta(a, b))
    >>> h  = entropy(Beta(a, b))
    >>> reward = intercept + slope * np.dot(stimulus, (mix_coef * mu  + (1 - mix_coef) * entropy))
    >>> # Observation is a normal random variable:
    >>> observation = Normal(reward, sigma)
    """
    name = 'BetaBinomial'

    def __init__(self, *args, a=1, b=1, sigma, mix_coef, intercept, slope, **kwargs):
        """
        Parameters
        ----------
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

        slope : float
            Slope used when computing the reward.

        Other Parameters
        ----------------
        **kwargs : any type
            All the mandatory keyword-only arguments required by :class:`ldmunit.models.associative_learning.base.CAMO` must also be
            provided during initialization.
        """
        assert a > 0, 'a must be positive'
        assert b > 0, 'b must be positive'
        assert sigma >= 0, 'sigma must be nonnegative'
        assert mix_coef >= 0 and mix_coef <= 1, 'mix_coef must be in range [0, 1]'
        paras = {'a': a, 'b': b, 'sigma': sigma, 'mix_coef': mix_coef, 'intercept': intercept, 'slope': slope}
        super().__init__(paras=paras, **kwargs)

    def _get_default_a_b(self):
        """
        Get default occurence and non-occurence counts.
        """
        a = self.paras['a']
        b = self.paras['b']
        out = {'a': a * np.ones(self.n_obs, dtype=np.float64), 'b': b * np.ones(self.n_obs, dtype=np.float64)}
        return out

    def reset(self):
        """
        Reset the hidden state to its default value.
        """
        self.hidden_state = DictWithBinarySequenceKeys()

    def observation(self, stimulus):
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
        assert self.observation_space.contains(stimulus)

        sd_pred = self.paras['sigma']

        mu_pred = self._predict_reward(stimulus)

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
        if stimulus not in self.hidden_state.keys():
            self.hidden_state[stimulus] = self._get_default_a_b()
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
        if stimulus not in self.hidden_state.keys():
            self.hidden_state[stimulus] = self._get_default_a_b()
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

        reward : float
            The reward for the action.

        done : bool
            If True, do not update the hidden state.
        """
        assert self.observation_space.contains(stimulus)
        # get model's state
        if stimulus not in self.hidden_state.keys():
            self.hidden_state[stimulus] = self._get_default_a_b()
        a = self.hidden_state[stimulus]['a']
        b = self.hidden_state[stimulus]['b']

        if not done:
            a += (1 - stimulus) * (1 - reward) + stimulus * reward
            b += (1 - stimulus) * reward + stimulus * (1 - reward)
            self.hidden_state[stimulus]['a'] = a
            self.hidden_state[stimulus]['b'] = b

        return a, b

    def _predict_reward(self, stimulus):
        """
        Predict the reward from the given stimulus using beta-binomial model
        equations.
        """
        mix_coef = self.paras['mix_coef']
        intercept = self.paras['intercept']
        slope = self.paras['slope']

        if tuple(stimulus) not in self.hidden_state.keys():
            self.hidden_state[stimulus] = self._get_default_a_b()
        a = self.hidden_state[stimulus]['a']
        b = self.hidden_state[stimulus]['b']

        mu = beta(a, b).mean()
        entropy = beta(a, b).entropy()

        rhat = intercept + slope * np.dot(stimulus, (mix_coef * mu + (1 - mix_coef) * entropy))

        return rhat
