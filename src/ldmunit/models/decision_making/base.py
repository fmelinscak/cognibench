from gym import spaces
from gym.utils import seeding
import numpy as np
from ...capabilities import DiscreteAction, DiscreteObservation
from .. import LDMModel

class DADO(LDMModel, DiscreteAction, DiscreteObservation):
    """
    Base class for models that operate on discrete action and discrete observation spaces.
    """

    def __init__(self, *args, n_action, n_obs, **kwargs):
        """
        Parameters
        ----------
        n_action : int
            Dimension of the action space.

        n_obs :
            Dimension of the observation space.
        """
        assert n_action > 0, 'n_action must be positive'
        assert n_obs > 0, 'n_obs must be positive'
        self.action_space = n_action 
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
            Each element of actions must contain one action. Further, each action
            be an array-like object whose length will be used as the dimension of the
            action space.
        """
        assert self._check_observation(stimuli) and self._check_action(actions)
        if not len(stimuli) == len(actions):
            raise AssertionError('stimuli and actions must be of the same length.')
        self.action_space = len(np.unique(actions))
        self.observation_space = len(np.unique(stimuli))
        print('action_space set to {}'.format(self.action_space))
        print('observation_space set to {}'.format(self.observation_space))
