import sciunit
from gym import spaces
import numpy as np
from gym.utils import seeding
from ...capabilities import Interactive, MultiBinaryObservation, Continuous, LogProbModel
from ...continuous import ContinuousSpace

class CAMO(sciunit.Model, MultiBinaryObservation, Continuous, Interactive, LogProbModel):

    def __init__(self, n_obs=None, paras=None, hidden_state=None, name=None, seed=None, **params):
        self.n_obs = n_obs
        self.paras = paras
        self.hidden_state = hidden_state
        self.seed = seed
        return super().__init__(n_obs=n_obs, paras=paras,
                                hidden_state=hidden_state, name=name, seed=seed, **params)
    @property
    def seed(self):
        return self._np_random

    @seed.setter
    def seed(self, value):
        if not value:
            self._np_random = None
        self._np_random, seed = seeding.np_random(value)

    @property
    def hidden_state(self):
        if self.n_obs and not self._hidden_state:
            self.reset()
        return self._hidden_state

    @hidden_state.setter
    def hidden_state(self, value):
        self._hidden_state = value

    def set_space_from_data(self, stimuli, actions):
        assert self._check_observation(stimuli) and self._check_action(actions)
        if not len(stimuli) == len(actions):
            raise AssertionError('stimuli and actions must be of the same length.')
        self.action_space = ContinuousSpace()
        self.observation_space = len(stimuli[0])

    @property
    def paras(self):
        return self._paras

    @paras.setter
    def paras(self, value):
        if not isinstance(value, dict) and value:
            raise TypeError("paras must be of dict type")
        else:
            self._paras = value
