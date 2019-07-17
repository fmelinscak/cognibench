import numpy as np
import gym
from gym import spaces
from scipy import stats
from scipy.stats import beta
from .base import CAMO
from ...capabilities import Interactive, LogProbModel

class BetaBinomialModel(CAMO, Interactive, LogProbModel):
    """
    Interactive beta-binomial model implementation.

    Occurence and non-occurence counts are stored in variables a and b, respectively.

    Reward is calculated as
    >>> mu = mean(Beta(a, b))
    >>> h  = entropy(Beta(a, b))
    >>> reward = intercept + slope * np.dot(stimulus, (mix_coef * mu  + (1 - mix_coef) * entropy))

    Observation is a normal random variable:
    >>> observation = Normal(reward, sigma)
    """
    name = 'BetaBinomial'

    def __init__(self, *args, a=1, b=1, sigma, mix_coef, intercept, slope, **kwargs):
        """
        Parameters
        ----------
        a : float
            Initial value of occurence count variable a. Must be positive.

        b : float
            Initial value of non-occurence count variable b. Must be positive.

        sigma : float
            Standard deviation of the normal distribution used to generate observations.
            Must be nonnegative.

        mix_coef : float
            Mixing coefficient used in the convex combination. Must be in [0, 1] range.

        intercept : float
            Intercept used when computing the reward.

        slope : float
            Slope used when computing the reward.
        """
        assert a > 0, 'a must be positive'
        assert b > 0, 'b must be positive'
        assert sigma >= 0, 'sigma must be nonnegative'
        assert mix_coef >= 0 and mix_coef <= 1, 'mix_coef must be in range [0, 1]'
        paras = {
            'a' : a,
            'b' : b,
            'sigma' : sigma,
            'mix_coef' : mix_coef,
            'intercept' : intercept,
            'slope' : slope
        }
        super().__init__(paras=paras, **kwargs)

    def _get_default_a_b(self):
        """
        Get default occurence and non-occurence counts.
        """
        a = self.paras['a']
        b = self.paras['b']
        out = {'a' : a * np.ones(self.n_obs, dtype=np.float64),
                'b' : b * np.ones(self.n_obs, dtype=np.float64)}
        return out

    def reset(self):
        """
        Reset the hidden state to its default value.
        """
        self.hidden_state = dict()

    def observation(self, stimulus):
        """
        Get the reward random variable for the given stimulus.

        Parameters
        ----------
        stimulus : array-like
            Single stimulus from the observation space.

        Returns
        -------
        scipy.stats.norm
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
        if tuple(stimulus) not in self.hidden_state.keys():
            self._set_hidden_state(stimulus, self._get_default_a_b())
        return self.observation(stimulus).logpdf

    def act(self, stimulus):
        if tuple(stimulus) not in self.hidden_state.keys():
            self._set_hidden_state(stimulus, self._get_default_a_b())
        return self.observation(stimulus).rvs()

    def update(self, stimulus, reward, action, done):
        assert self.observation_space.contains(stimulus)
        # get model's state
        if tuple(stimulus) not in self.hidden_state.keys():
            self._set_hidden_state(stimulus, self._get_default_a_b())
        a = self._get_hidden_state(stimulus)['a']
        b = self._get_hidden_state(stimulus)['b']

        if not done:
            a += (1 - stimulus) * (1 - reward) + stimulus * reward
            b += (1 - stimulus) * reward + stimulus * (1 - reward)
            old = self._get_hidden_state(stimulus)
            old['a'] = a
            old['b'] = b
            self._set_hidden_state(stimulus, old)

        return a, b

    def _set_hidden_state(self, key, val):
        self.hidden_state[tuple(key)] = val

    def _get_hidden_state(self, key):
        return self.hidden_state[tuple(key)]

    def _predict_reward(self, stimulus):
        """
        Predict the reward from the given stimulus using beta-binomial model
        equations.
        """
        mix_coef = self.paras['mix_coef']
        intercept = self.paras['intercept']
        slope = self.paras['slope']

        if tuple(stimulus) not in self.hidden_state.keys():
            self._set_hidden_state(stimulus, self._get_default_a_b())
        a = self._get_hidden_state(stimulus)['a']
        b = self._get_hidden_state(stimulus)['b']

        mu      = beta(a,b).mean()
        entropy = beta(a,b).entropy()

        rhat = intercept + slope * np.dot(stimulus, (mix_coef * mu  + (1 - mix_coef) * entropy))
        
        return rhat
