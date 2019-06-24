import sciunit
import numpy as np
import gym
from gym import spaces
from scipy import stats

# import oct2py
# from oct2py import Struct
# import inspect
# import os

from ...capabilities import Interactive, ContinuousAction, MultibinObsevation

class RwNormModel(sciunit.Model, Interactive, ContinuousAction, MultibinObsevation):
    
    action_space = spaces.Box
    observation_space = spaces.MultiBinary

    def __init__(self, n_obs, paras=None, name=None):
        assert isinstance(n_obs, int)
        self.paras = paras
        self.name = name
        self.n_obs = n_obs
        self.hidden_state = self._set_hidden_state()
        self.action_space = self._set_action_space()
        self.observation_space = self._set_observation_space(self.n_obs)

    def _set_hidden_state(self):
        w0 = 0
        if 'w0' in self.paras:
            w0 = self.paras['w0']
        hidden_state = {'w': np.full(self.n_obs, w0)}
        return hidden_state

    def _set_observation_space(self, n_obs):
        return spaces.MultiBinary(n_obs)

    def _set_action_space(self):
        return spaces.Box(-1000, 1000, shape=(1,), dtype=np.float32)

    def reset(self):
        self.action_space = self._set_action_space()
        self.observation_space = self._set_observation_space(self.n_obs)

    def predict(self, stimulus):
        assert self.observation_space.contains(stimulus)

        sd_pred = self.paras['sigma']
        mu_pred = self.act(stimulus)

        return stats.norm(loc=mu_pred[0], scale=sd_pred).logpdf

    def update(self, stimulus, reward, action, done):
        assert self.action_space.contains(action)
        assert self.observation_space.contains(stimulus)

        alpha  = self.paras['alpha']
        w_curr = self.hidden_state['w']
        pred_reward = self.act(stimulus)[0]

        if not done:
            pred_err = reward - pred_reward

            w_curr += alpha * pred_err * stimulus

            self.hidden_state['w'] = w_curr

        return w_curr
    
    def act(self, stimulus):
        assert self.observation_space.contains(stimulus)

        b0 = self.paras['b0'] # intercept
        b1 = self.paras['b1'] # slope #TODO: add variable slopes
        
        w_curr = self.hidden_state['w']

        # Generate reward prediction
        rhat = np.dot(stimulus, w_curr.T)

        # Predict response
        mu_pred = b0 + b1 * rhat
        return np.array([mu_pred])

# class RwNormOctModel(RwNormModel):
    
#     def update(self, stimulus, reward, action, done):
#         """observation function"""
#         assert self.action_space.contains(action)
#         assert self.observation_space.contains(stimulus)

#         class_path = os.path.dirname(inspect.getfile(type(self))) #TODO: fix this 

#         params = Struct()
#         params['alpha'] = self.paras['alpha'] # slope
#         params['wInit'] = self.hidden_state['w']

#         # Calculate predictions in Octave
#         with oct2py.Oct2Py() as oc:
#             oc.addpath(class_path)
#             result = oc.evo_rw_batch(stimulus, reward, params)
        
#         self.hidden_state['w'] = result.w

#         return result.w
    
#     def act(self, stimulus):
#         """observation function"""
#         assert self.observation_space.contains(stimulus)

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