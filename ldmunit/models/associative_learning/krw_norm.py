import numpy as np
import gym
from gym import spaces
from scipy import stats
from .base import CAMO
from ...capabilities import Interactive, LogProbModel
from ...utils import is_arraylike

class KrwNormModel(CAMO, Interactive, LogProbModel):
    """
    Kalman Rescorla-Wagner model implementation.
    """
    name = 'KrwNorm'

    def __init__(self, *args, w, sigma, b0, b1, sigmaWInit, tauSq, sigmaRSq, **kwargs):
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

        sigmaWInit : float
            Diagonal elements of the covariance matrix C is set to sigmaWInit.

        tauSq : float
            Diagonal elements of the transition noise variance matrix Q is set to tauSq.
            Must be nonnegative.

        sigmaRSq : float
            Additive factor used in the denominator when computing the Kalman gain K.
            Must be nonnegative.

        Other Parameters
        ----------------
        **kwargs : any type
            All the mandatory keyword-only arguments required by :class:`ldmunit.models.associative_learning.base.CAMO` must also be
            provided during initialization.
        """
        assert sigma >= 0, 'sigma must be nonnegative'
        assert tauSq >= 0, 'tauSq must be nonnegative'
        assert sigmaRSq >= 0, 'tauSq must be nonnegative'
        paras = {
            'w' : w,
            'sigma' : sigma,
            'b0' : b0,
            'b1' : b1,
            'sigmaWInit' : sigmaWInit,
            'tauSq' : tauSq,
            'sigmaRSq' : sigmaRSq
        }
        super().__init__(paras=paras, **kwargs)
        if is_arraylike(w):
            assert len(w) == self.n_obs, 'w must have the same length as the dimension of the observation space'

    def reset(self):
        """
        Reset the hidden state to its default value.
        """
        w = self.paras['w'] if 'w' in self.paras else 0
        if is_arraylike(w):
            w = np.array(w, dtype=np.float64)
        else:
            w = np.full(self.n_obs, w, dtype=np.float64)

        sigmaWInit = self.paras['sigmaWInit']
        C =  sigmaWInit * np.identity(self.n_obs)

        self.hidden_state = {'w': np.full(self.n_obs, w),
                             'C': C}

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
        :class:`scipy.stats.rv_continuous`
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
        assert self.action_space.contains(action)
        assert self.observation_space.contains(stimulus)

        tauSq = self.paras['tauSq'] # State diffusion variance
        Q = tauSq * np.identity(self.n_obs) # Transition noise variance (transformed to positive reals); constant over time
        sigmaRSq = self.paras['sigmaRSq']

        
        w_curr = self.hidden_state['w']
        C_curr = self.hidden_state['C']

        rhat = self._predict_reward(stimulus)

        if not done:                
            # Kalman prediction step
            w_pred = w_curr # No mean-shift for the weight distribution evolution (only stochastic evolution)
            C_pred = C_curr + Q # Update covariance

            # get pred_error
            delta = reward - rhat

            # Kalman update step
            K = C_pred.dot(stimulus) / (stimulus.dot(C_pred.dot(stimulus)) + sigmaRSq) # (n_obs,)
            w_updt = w_pred + K * delta # Mean updated with prediction error
            C_updt = C_pred - K * stimulus * C_pred # Covariance updated

            self.hidden_state['w'] = w_updt
            self.hidden_state['C'] = C_updt

        return w_updt, C_updt
