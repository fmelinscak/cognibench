import sciunit
import numpy as np
import gym
from gym import spaces
from scipy.optimize import minimize
from scipy import stats
from ..capabilities import Interactive

class KrwNormModel(sciunit.Model, Interactive):
    
    action_space = spaces.Box
    observation_space = spaces.MultiBinary

    def __init__(self, n_obs, paras=None, name=None):
        assert isinstance(n_obs, int)
        self.paras = paras
        self.name = name
        self.n_obs = n_obs
        self._set_spaces(n_obs)
        self.hidden_state = self._set_hidden_state(n_obs)

    def _set_hidden_state(self, n_obs):
        w0 = 0
        if 'w0' in self.paras:
            w0 = self.paras['w0']
        w0 = np.array(w0) if isinstance(w0, list) else np.full(n_obs, w0)

        logSigmaWInit = self.paras['logSigmaWInit']
        C =  np.exp(logSigmaWInit) * np.identity(self.n_obs) # Initial weight covariance matrix

        hidden_state = {'w': np.full(n_obs, w0),
                        'C': C}
        return hidden_state

    def _set_spaces(self, n_obs):
        self.action_space = spaces.Box(-1000, 1000, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.MultiBinary(n_obs)

    def predict(self, stimulus):
        assert self.observation_space.contains(stimulus)

        sd_pred = self.paras['sigma']
        mu_pred = self.act(stimulus)

        return stats.norm(loc=mu_pred, scale=sd_pred).logpdf

    def update(self, stimulus, reward, action, done):
        """evolution function"""
        assert self.observation_space.contains(stimulus)

        alpha  = self.paras['alpha']
        tauSq = np.exp(self.paras['logTauSq']) # State diffusion variance
        Q = tauSq * np.identity(self.n_obs) # Transition noise variance (transformed to positive reals); constant over time
        sigmaRSq = np.exp(self.paras['logSigmaRSq'])

        
        w_curr = self.hidden_state['w']
        C_curr = self.hidden_state['C']

        if not done:
            # Kalman prediction step
            w_pred = w_curr # No mean-shift for the weight distribution evolution (only stochastic evolution)
            C_pred = C_curr + Q # Update covariance

            # get pred_error
            pred_err = reward - action

            # Kalman update step
            K = C_pred.dot(stimulus) / (stimulus.dot(C_pred.dot(stimulus)) + sigmaRSq) # (n_obs,)
            w_updt = w_pred + K * pred_err # Mean updated with prediction error
            C_updt = C_pred - K * stimulus * C_pred # Covariance updated

            self.hidden_state['w'] = w_updt
            self.hidden_state['C'] = C_updt

        return w_updt, C_updt

    def reset(self):
        self.hidden_state = self._set_hidden_state(self.n_obs)
        return None
    
    def act(self, stimulus):
        """observation function"""
        b0 = self.paras['b0'] # intercept
        b1 = self.paras['b1'] # slope #TODO: add variable slopes
        
        w_curr = self.hidden_state['w']

        # Generate reward prediction
        rhat = np.dot(stimulus, w_curr.T)

        # Predict response
        mu_pred = b0 + b1 * rhat
        return mu_pred