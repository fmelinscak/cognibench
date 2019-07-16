import numpy as np
import gym
from gym import spaces
from scipy import stats
from .base import CAMO
from ...capabilities import Interactive, LogProbModel

class KrwNormModel(CAMO, Interactive, LogProbModel):
    name = 'KrwNorm'

    def __init__(self, *args, w0, sigma, b0, b1, logSigmaWInit, logTauSq, logSigmaRSq, **kwargs):
        paras = {
            'w0' : w0,
            'sigma' : sigma,
            'b0' : b0,
            'b1' : b1,
            'logSigmaWInit' : logSigmaWInit,
            'logTauSq' : logTauSq,
            'logSigmaRSq' : logSigmaRSq
        }
        super().__init__(paras=paras, **kwargs)

    def reset(self):
        w0 = self.paras['w0'] if 'w0' in self.paras else 0
        try:
            it = iter(w0)
            w0 = np.array(w0, dtype=np.float64)
        except TypeError:
            w0 = np.full(self.n_obs, w0, dtype=np.float64)

        logSigmaWInit = self.paras['logSigmaWInit']
        C =  np.exp(logSigmaWInit) * np.identity(self.n_obs) # Initial weight covariance matrix

        self.hidden_state = {'w': np.full(self.n_obs, w0),
                             'C': C}

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

        tauSq = np.exp(self.paras['logTauSq']) # State diffusion variance
        Q = tauSq * np.identity(self.n_obs) # Transition noise variance (transformed to positive reals); constant over time
        sigmaRSq = np.exp(self.paras['logSigmaRSq'])

        
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
