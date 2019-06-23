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

class RandomRespondModel(sciunit.Model, Interactive):
    
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
        return hidden_state

    def _set_spaces(self):
        self.action_space = spaces.Box(-1000, 1000, shape=(1,), dtype=np.float32) #TODO, to be changed
        self.observation_space = spaces.MultiBinary(self.n_obs)

    def predict(self, stimulus):
        assert self.observation_space.contains(stimulus)
        pass

    def update(self, stimulus, reward, action, done):
        assert self.action_space.contains(action)
        assert self.observation_space.contains(stimulus)
        pass

    def reset(self):
        self.hidden_state = self._set_hidden_state()
        return None
    
    def act(self, stimulus):
        assert self.observation_space.contains(stimulus)
        pass
