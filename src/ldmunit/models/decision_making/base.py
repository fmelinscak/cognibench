import sciunit
from gym import spaces
from gym.utils import seeding
import numpy as np
from ...capabilities import Interactive, DiscreteAction, DiscreteObservation

class DADO(sciunit.Model, DiscreteAction, DiscreteObservation, Interactive):

    def __init__(self, n_action=None, n_obs=None, paras=None, hidden_state=None, seed=None, name=None, **params):
        self.n_action = n_action
        self.n_obs = n_obs
        self.paras = paras
        self.hidden_state = hidden_state
        self.seed = seed
        return super().__init__(n_action=n_action, n_obs=n_obs, paras=paras, seed=seed,
                                hidden_state=hidden_state, name=name, **params)

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
        self.action_space = len(np.unique(actions))
        self.observation_space = len(np.unique(stimuli))
        print('action_space set to {}'.format(self.action_space))
        print('observation_space set to {}'.format(self.observation_space))

    @property
    def paras(self):
        return self._paras

    @paras.setter
    def paras(self, value):
        if not isinstance(value, dict) and value:
            raise TypeError("paras must be of dict type")
        else:
            self._paras = value