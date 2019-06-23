import sciunit
import numpy as np
from gym import spaces
from scipy import stats
from ...capabilities import Interactive

class RandomRespondModel(sciunit.Model, Interactive):
    """Random respond for discrete decision marking."""

    action_space = spaces.Discrete
    observation_space = spaces.Discrete

    def __init__(self, n_actions, n_obs, paras=None, name=None):
        self.paras = paras
        self.name = name
        self.n_actions = n_actions
        self.n_obs = n_obs
        self._set_spaces()
        self.hidden_state = self._set_hidden_state()
        
    def _set_hidden_state(self):
        # set rv_discrete for each stimulus/cue/observation
        bias        = self.paras['bias']
        action_bias = self.paras['action_bias']
        pk = np.full(self.n_actions, (1 - bias) / (self.n_actions - 1))
        pk[action_bias] = bias

        xk = np.arange(self.n_actions)
        rv = stats.rv_discrete(name=self.name, values=(xk, pk))

        hidden_state = {'rv': dict([[i, rv] for i in range(self.n_obs)])}
        return hidden_state

    def _set_spaces(self):
        self.action_space = spaces.Discrete(self.n_actions)
        self.observation_space = spaces.Discrete(self.n_actions)

    def predict(self, stimulus):
        assert self.observation_space.contains(stimulus)
        rv = self.hidden_state['rv'][stimulus]
        return rv.logpmf

    def update(self, stimulus, reward, action, done):
        assert self.action_space.contains(action)
        assert self.observation_space.contains(stimulus)
        pass

    def reset(self):
        self.hidden_state = self._set_hidden_state()
        return None
    
    def act(self, stimulus):
        assert self.observation_space.contains(stimulus)
        rv = self.hidden_state['rv'][stimulus]
        return rv.rvs()

