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

class RandomRespondModel(sciunit.Model, Interactive):
    
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
        return None

    def _set_observation_space(self, n_obs):
        return spaces.MultiBinary(n_obs)

    def _set_action_space(self):
        return spaces.Box(-1000, 1000, shape=(1,), dtype=np.float32)

    def reset(self):
        self.action_space = self._set_action_space()
        self.observation_space = self._set_observation_space(self.n_obs)

    def predict(self, stimulus):
        assert self.observation_space.contains(stimulus)
        pass

    def update(self, stimulus, reward, action, done):
        assert self.action_space.contains(action)
        assert self.observation_space.contains(stimulus)
        pass
    
    def act(self, stimulus):
        assert self.observation_space.contains(stimulus)
        pass
