import sciunit
import numpy as np
import gym
from gym import spaces
from scipy import stats
from scipy.stats import beta

# import oct2py
# from oct2py import Struct
# import inspect
# import os

from ...capabilities import Interactive, ContinuousAction, MultibinObsevation

class BetaBinomialModel(sciunit.Model, Interactive, ContinuousAction, MultibinObsevation):

    action_space = spaces.Box
    observation_space = spaces.MultiBinary

    def __init__(self, n_obs, paras=None, name=None):
        self.paras = paras
        self.name = name
        self.n_obs = n_obs
        self.hidden_state = self._set_hidden_state()
        self.action_space = self._set_action_space()
        self.observation_space = self._set_observation_space(self.n_obs)

    def _set_hidden_state(self):
        hidden_state = {'a' : np.ones(self.n_obs),
                        'b' : np.ones(self.n_obs)}
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
        # get model's state
        a = self.hidden_state['a']
        b = self.hidden_state['b']

        logpdf = lambda x: beta.logpdf(x, a=a, b=b)
        return logpdf

    def update(self, stimulus, reward, action, done): #TODO: add default value
        assert self.observation_space.contains(stimulus)
        # get model's state
        a = self.hidden_state['a']
        b = self.hidden_state['b']

        if not done:
            a += (1 - stimulus) * (1 - reward) + stimulus * reward
            b += (1 - stimulus) * reward + stimulus * (1 - reward)
            self.hidden_state['a'] = a
            self.hidden_state['b'] = b

        return a, b

    def act(self, stimulus):
        """observation function"""
        assert self.observation_space.contains(stimulus)

        mix_coef = self.paras['mix_coef']
        b0 = self.paras['b0'] # intercept
        b1 = self.paras['b1'] # slope #TODO: add variable slopes

        # get model's state
        a = self.hidden_state['a']
        b = self.hidden_state['b']

        # Generate outcome prediction
        mu      = beta.mean(a, b)
        entropy = beta.entropy(a, b)

        crPred = b0 + np.dot(stimulus, (mix_coef * mu  + (1 - mix_coef) * entropy)) * b1
        
        return np.array([crPred])
