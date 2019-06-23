import sciunit
import numpy as np
import gym
from gym import spaces
from scipy import stats

# import oct2py
# from oct2py import Struct
# import inspect
# import os

from ...capabilities import Interactive

class KrwNormModel(sciunit.Model, Interactive):
    
    action_space = spaces.Box
    observation_space = spaces.MultiBinary

    def __init__(self, n_obs, paras=None, name=None):
        assert isinstance(n_obs, int)
        self.paras = paras
        self.name = name
        self.n_obs = n_obs
        self._set_spaces()
        self.hidden_state = self._set_hidden_state()

    def _set_hidden_state(self):
        w0 = 0
        if 'w0' in self.paras:
            w0 = self.paras['w0']
        w0 = np.array(w0) if isinstance(w0, list) else np.full(self.n_obs, w0)

        logSigmaWInit = self.paras['logSigmaWInit']
        C =  np.exp(logSigmaWInit) * np.identity(self.n_obs) # Initial weight covariance matrix

        hidden_state = {'w': np.full(self.n_obs, w0),
                        'C': C}
        return hidden_state

    def _set_spaces(self):
        self.action_space = spaces.Box(-1000, 1000, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.MultiBinary(self.n_obs)

    def predict(self, stimulus):
        assert self.observation_space.contains(stimulus)

        sd_pred = self.paras['sigma']
        mu_pred = self.act(stimulus)

        return stats.norm(loc=mu_pred[0], scale=sd_pred).logpdf

    def update(self, stimulus, reward, action, done):
        """evolution function"""
        assert self.action_space.contains(action)
        assert self.observation_space.contains(stimulus)

        tauSq = np.exp(self.paras['logTauSq']) # State diffusion variance
        Q = tauSq * np.identity(self.n_obs) # Transition noise variance (transformed to positive reals); constant over time
        sigmaRSq = np.exp(self.paras['logSigmaRSq'])

        
        w_curr = self.hidden_state['w']
        C_curr = self.hidden_state['C']

        pred_reward = self.act(stimulus)[0]

        if not done:
            # Kalman prediction step
            w_pred = w_curr # No mean-shift for the weight distribution evolution (only stochastic evolution)
            C_pred = C_curr + Q # Update covariance

            # get pred_error
            pred_err = reward - pred_reward

            # Kalman update step
            K = C_pred.dot(stimulus) / (stimulus.dot(C_pred.dot(stimulus)) + sigmaRSq) # (n_obs,)
            w_updt = w_pred + K * pred_err # Mean updated with prediction error
            C_updt = C_pred - K * stimulus * C_pred # Covariance updated

            self.hidden_state['w'] = w_updt
            self.hidden_state['C'] = C_updt

        return w_updt, C_updt

    def reset(self):
        self.hidden_state = self._set_hidden_state()
        return None
    
    def act(self, stimulus):
        """observation function"""
        assert self.observation_space.contains(stimulus)

        b0 = self.paras['b0'] # intercept
        b1 = self.paras['b1'] # slope #TODO: add variable slopes
        
        w_curr = self.hidden_state['w']

        # Generate reward prediction
        rhat = np.dot(stimulus, w_curr.T)

        # Predict response
        mu_pred = b0 + b1 * rhat
        return [mu_pred]

# class KrwNormOctModel(KrwNormModel):
    
#     def update(self, stimulus, reward, action, done):
#         """observation function"""
#         assert self.action_space.contains(action)
#         assert self.observation_space.contains(stimulus)

#         class_path = os.path.dirname(inspect.getfile(type(self))) #TODO: fix this 

#         params = Struct()
#         params['w'] = self.hidden_state['w']
#         params['logTauSq'] = self.paras['logTauSq']
#         params['logSigmaRSq'] = self.paras['logSigmaRSq']
#         params['C'] = self.hidden_state['C']

#         # Calculate predictions in Octave
#         with oct2py.Oct2Py() as oc:
#             oc.addpath(class_path)
#             result = oc.evo_krw_batch(stimulus, reward, params)
        
#         self.hidden_state['w'] = result.w
#         self.hidden_state['C'] = result.C

#         return result.w, result.C
    
#     def act(self, stimulus):
#         """observation function"""
#         assert self.observation_space.contains(stimulus)

#         # same as the rw_norm
#         class_path = os.path.dirname(inspect.getfile(type(self))) #TODO: fix this 

#         params = Struct()
#         params['slope']     = self.paras['b1'] # slope
#         params['intercept'] = self.paras['b0'] # intercept

#         results_evo = Struct()
#         results_evo['w'] = self.hidden_state['w']

#         # Calculate predictions in Octave
#         with oct2py.Oct2Py() as oc:
#             oc.addpath(class_path)
#             result = oc.obs_affine_batch(results_evo, stimulus, params)
        
#         return result.crPred