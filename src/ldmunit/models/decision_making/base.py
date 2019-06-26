import sciunit
from gym import spaces
import numpy as np
from ...capabilities import Interactive, DiscreteAction, DiscreteObservation

class DADO(sciunit.Model, DiscreteAction, DiscreteObservation):

    def __init__(self, n_action=None, n_obs=None, paras=None, hidden_state=None, name=None, **params):
        self.n_action = n_action
        self.n_obs = n_obs
        self.paras = paras
        self.hidden_state = hidden_state
        return super().__init__(n_action=n_action, n_obs=n_obs, paras=paras,
                                hidden_state=hidden_state, name=name, **params)

    @property
    def hidden_state(self):
        return self._hidden_state

    @hidden_state.setter
    def hidden_state(self, value):
        if not value:
            try:
                self.reset()
            except TypeError:
                self._hidden_state = None
        else:
            self._hidden_state = value

    def set_space_from_data(self, stimuli, actions):
        assert self._check_observation(stimuli) and self._check_action(actions)
        if not len(stimuli) == len(actions):
            raise AssertionError('stimuli and actions must be of the same length.')
        self.action_space = len(np.unique(actions))
        self.observation_space = len(np.unique(stimuli))

    @property
    def paras(self):
        return self._paras

    @paras.setter
    def paras(self, value):
        if not value:
            self._paras = self._get_default_paras()
        elif not isinstance(value, dict):
            raise TypeError("paras must be of dict type")
        else:
            self._paras = value

    def _get_default_paras(self):
        raise NotImplementedError("Must implement _get_default_paras.")
        
    def reset(self):
        raise NotImplementedError("Must implement reset.")

    def predict(self):
        raise NotImplementedError("Must implement predict.")

    def update(self):
        raise NotImplementedError("Must implement update.")

    def act(self):
        raise NotImplementedError("Must implement act.")
