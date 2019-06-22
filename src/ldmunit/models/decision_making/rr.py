import sciunit
import numpy as np
from gym import spaces
from scipy.optimize import minimize
from scipy import stats
from ...capabilities import Interactive

class RRModel(sciunit.Model, Interactive):
    """Random respounding for discrete decision marking."""

    action_space = spaces.Discrete
    observation_space = spaces.Discrete

    def __init__(self, n_actions, n_obs, paras=None, name=None):
        self.paras = paras
        self.name = name
        self.n_actions = n_actions
        self.n_obs = n_obs
        self._set_spaces(n_actions)
        self.hidden_state = self._set_hidden_state(n_actions, n_obs)
        
    def _set_hidden_state(self, n_actions, n_obs):
        # set rv_discrete for each stimulus/cue/observation
        xk = np.arange(n_actions)
        pk = np.full(n_actions, 1/n_actions)
        rv = stats.rv_discrete(name=self.name, values=(xk, pk))

        hidden_state = {'RV': dict([[i, rv] for i in range(n_obs)])}
        return hidden_state

    def _set_spaces(self, n_actions):
        self.action_space = spaces.Discrete(n_actions)
        self.observation_space = spaces.Discrete(n_actions)

    def predict(self, stimulus):
        assert self.observation_space.contains(stimulus)
        # get model's state
        RV = self.hidden_state['RV'][stimulus]
        
        return RV.logpmf

    def update(self, stimulus, reward, action, done): #TODO: add default value
        assert self.action_space.contains(action)
        assert self.observation_space.contains(stimulus)
        pass

    def reset(self):
        self.hidden_state = self._set_hidden_state(self.n_actions, self.n_obs)
        return None
    
    def act(self, stimulus):
        RV = self.hidden_state['RV'][stimulus]
        return RV.rvs()

