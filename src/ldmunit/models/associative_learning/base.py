from gym import spaces
import numpy as np
from gym.utils import seeding
from ...capabilities import MultiBinaryObservation, ContinuousAction
from ...continuous import ContinuousSpace
from .. import LDMModel

class CAMO(LDMModel, ContinuousAction, MultiBinaryObservation):
    """
    Base class for models that operate on continuous action and multi-binary observation spaces.
    """

    def __init__(self, *args, n_obs, **kwargs):
        """
        n_obs : int
            Dimension of the observation space.
        """
        self.observation_space = n_obs
        super().__init__(**kwargs)

    def set_space_from_data(self, stimuli, actions):
        assert self._check_observation(stimuli) and self._check_action(actions)
        if not len(stimuli) == len(actions):
            raise AssertionError('stimuli and actions must be of the same length.')
        self.action_space = ContinuousSpace()
        self.observation_space = len(stimuli[0])

