import numpy as np
import gym
from gym import spaces
from scipy import stats
from .base import CAMO
from ...capabilities import Interactive, LogProbModel
from ...utils import is_arraylike

class RwNormModel(CAMO, Interactive, LogProbModel):
    """
    Rescorla-Wagner model implementation.
    """
    name = 'RwNorm'

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

        b1 : float
            Slope used when computing the mean of the normal distribution from reward.

        eta : float
            Learning rate for w updates. Must be nonnegative.
        """
        assert sigma >= 0, 'sigma must be nonnegative'
        assert eta >= 0, 'eta must be nonnegative'
        paras = {
            'w' : w,
            'b0' : b0,
            'b1' : b1,
            'sigma' : sigma,
            'eta' : eta
        }
        super().__init__(paras=paras, **kwargs)
        if is_arraylike(w):
            assert len(w) == self.n_obs, 'w must have the same length as the dimension of the observation space'

    def reset(self):
        w = self.paras['w'] if 'w' in self.paras else 0
        if is_arraylike(w):
            w = np.array(w, dtype=np.float64)
        else:
            w = np.full(self.n_obs, w, dtype=np.float64)

        self.hidden_state = {'w': w}

    def predict(self, stimulus):
        return self.observation(stimulus).logpdf

    def act(self, stimulus):
        return self.observation(stimulus).rvs()

    def _predict_reward(self, stimulus):
        assert self.observation_space.contains(stimulus)
        w_curr = self.hidden_state['w']
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
        scipy.stats.norm
            Normal random variable with mean equal to linearly transformed
            reward using b0 and b1 parameters, and standard deviation equal
            to sigma model parameter.
        """
        assert self.hidden_state, "hidden state must be set"
        assert self.observation_space.contains(stimulus)

        b0 = self.paras['b0'] # intercept
        b1 = self.paras['b1'] # slope
        sd_pred = self.paras['sigma']
        
        w_curr = self.hidden_state['w']

        rhat = self._predict_reward(stimulus)

        # Predict response
        mu_pred = b0 + b1 * rhat

        rv = stats.norm(loc=mu_pred, scale=sd_pred)
        rv.random_state = self.seed

        return rv

    def update(self, stimulus, reward, action, done):
        assert self.action_space.contains(action)
        assert self.observation_space.contains(stimulus)

        eta  = self.paras['eta']
        w_curr = self.hidden_state['w']

        rhat = self._predict_reward(stimulus)

        if not done:
            delta = reward - rhat
            w_curr += eta * delta * stimulus
            self.hidden_state['w'] = w_curr

        return w_curr
