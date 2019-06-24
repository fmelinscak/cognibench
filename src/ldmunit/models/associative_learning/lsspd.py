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

class LSSPDModel(sciunit.Model, Interactive, ContinuousAction, MultibinObsevation):
    
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
        #TODO: write a wrapper function for paras
        w0 = 0
        if 'w0' in self.paras:
            w0 = self.paras['w0']

        alpha = 0
        if 'alpha' in self.paras:
            alpha = self.paras['alpha']

        w0 = np.array(w0) if isinstance(w0, list) else np.full(self.n_obs, w0)
        alpha = np.array(alpha) if isinstance(alpha, list) else np.full(self.n_obs, alpha)

        hidden_state = {'w'    : w0,
                        'alpha': alpha}
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
        """evolution function"""
        assert self.action_space.contains(action)
        assert self.observation_space.contains(stimulus)

        eta   = self.paras['eta'] # Proportion of pred. error. in the updated associability value
        kappa = self.paras['kappa'] # Fixed learning rate for the cue weight update
        
        w_curr = self.hidden_state['w']
        alpha  = self.hidden_state['alpha']

        pred_reward = self.act(stimulus)[0]


        if not done:
            pred_err = reward - pred_reward

            w_curr += kappa * pred_err * alpha * stimulus # alpha, stimulus size: (n_obs,)

            # if stimulus[i] = 1
            # alpha[i] = eta * abs(pred_err) + (1 - eta) * alpha[i]
            # or
            # alpha[i] -= eta * alpha[i]
            # alpha[i] += eta * abs(pred_err)
            alpha -= eta * np.multiply(alpha, stimulus)
            alpha += eta * abs(pred_err) * stimulus
            np.clip(alpha, a_min=0, a_max=1, out=alpha) # Enforce upper bound on alpha

            self.hidden_state['w'] = w_curr
            self.hidden_state['alpha'] = alpha

        return w_curr, alpha

    def act(self, stimulus):
        """observation function"""
        assert self.observation_space.contains(stimulus)

        b0 = self.paras['b0'] # intercept
        b1 = self.paras['b1'] # slope #TODO: add variable slopes
        mix_coef = self.paras['mix_coef'] # proportion of the weights signal in the mixture of weight and associability signals
        
        w_curr = self.hidden_state['w']
        alpha = self.hidden_state['alpha']

        # Predict response
        mu_pred = b0 + b1 * np.dot(stimulus, (mix_coef * w_curr + (1 - mix_coef) * alpha))
        return np.array([mu_pred])

class LSSPDOctModel(LSSPDModel):
    
    def update(self, stimulus, reward, action, done):
        """observation function"""
        assert self.action_space.contains(action)
        assert self.observation_space.contains(stimulus)

        pass
    
    def act(self, stimulus):
        """observation function"""
        assert self.observation_space.contains(stimulus)

        pass