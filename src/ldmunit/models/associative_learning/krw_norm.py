import sciunit
import numpy as np
import gym
from gym import spaces
from scipy import stats
from .rw_norm import RwNormModel

class KrwNormModel(RwNormModel):

    def __init__(self, n_obs=None, paras=None, hidden_state=None, name=None, **params):
        return super().__init__(n_obs=n_obs, paras=paras, hidden_state=hidden_state, name=name, **params)
        
    def reset(self):
        w0 = self.paras['w0'] if 'w0' in self.paras else 0
        w0 = np.array(w0, dtype=np.float64) if isinstance(w0, list) else np.full(self.n_obs, w0, dtype=np.float64)

        logSigmaWInit = self.paras['logSigmaWInit']
        C =  np.exp(logSigmaWInit) * np.identity(self.n_obs) # Initial weight covariance matrix

        hidden_state = {'w': np.full(self.n_obs, w0),
                        'C': C}
        self.hidden_state = hidden_state

    def _get_default_paras(self):
        return {'w0': 0.1, 'sigma': 0.5, 'b0': 0.5, 'b1': 0.5, 'logSigmaWInit': 0.23, 'logTauSq': 0.32, 'logSigmaRSq': 0.34}
        
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
