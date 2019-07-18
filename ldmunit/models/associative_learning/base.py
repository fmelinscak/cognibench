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
        Parameters
        ----------
        n_obs : int
            Dimension of the multi-binary observation space. For example, when n_obs is 4,
            [0, 1, 1, 0] is a possible sample from the observation space since it consists of
            4 binary values. Must be positive.
        """
        assert n_obs > 0, 'n_obs must be positive'
        self.observation_space = n_obs
        super().__init__(**kwargs)

    def set_space_from_data(self, stimuli, actions):
        """
        Infer action and observation spaces from given simulation data.

        Parameters
        ----------
        stimuli : array-like
            Each element of stimuli must contain one stimulus. Further, each stimulus
            be an array-like object whose length will be used as the dimension of the
            observation space.

        actions : array-like
            Each element of actions must contain a continuous action. Lengths of
            stimuli and actions must be the same.
        """
        assert self._check_observation(stimuli) and self._check_action(actions)
        if not len(stimuli) == len(actions):
            raise AssertionError('stimuli and actions must be of the same length.')
        self.action_space = ContinuousSpace()
        self.observation_space = len(stimuli[0])
