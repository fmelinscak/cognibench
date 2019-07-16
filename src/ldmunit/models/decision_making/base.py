from gym import spaces
from gym.utils import seeding
import numpy as np
from ...capabilities import DiscreteAction, DiscreteObservation
from .. import LDMModel

class DADO(LDMModel, DiscreteAction, DiscreteObservation):
    def __init__(self, *args, n_action, n_obs, **kwargs):
        self.action_space = n_action 
        self.observation_space = n_obs
        super().__init__(**kwargs)

    def set_space_from_data(self, stimuli, actions):
        assert self._check_observation(stimuli) and self._check_action(actions)
        if not len(stimuli) == len(actions):
            raise AssertionError('stimuli and actions must be of the same length.')
        self.action_space = len(np.unique(actions))
        self.observation_space = len(np.unique(stimuli))
        print('action_space set to {}'.format(self.action_space))
        print('observation_space set to {}'.format(self.observation_space))
