import sciunit
import numpy as np
from gym import spaces
from scipy.optimize import minimize
from scipy import stats
from ...capabilities import Interactive

class NWSLSModel(sciunit.Model, Interactive):
    """Noisy-win-stay-lose-shift model"""

    action_space = spaces.Discrete
    observation_space = spaces.Discrete

    def __init__(self, n_actions, n_obs, paras=None, name=None):
        self.paras = paras
        self.name = name
        self.n_actions = n_actions
        self.n_obs = n_obs # number of stimuli
        self._set_spaces(n_actions)
        self.hidden_state = self._set_hidden_state()

    def _set_hidden_state(self):
        hidden_state = {'pk': dict([[i, np.full(self.n_actions, 1 / self.n_actions)] 
                                   for i in range(self.n_obs)])}
        return hidden_state

    def _set_spaces(self, n_actions):
        self.action_space = spaces.Discrete(n_actions)
        self.observation_space = spaces.Discrete(n_actions)

    def predict(self, stimulus):
        assert self.observation_space.contains(stimulus)
        self._rv.pk = self.hidden_state['pk'][stimulus]
        return self._rv.logpmf

    def update(self, stimulus, reward, action, done):
        pk = self.hidden_state['pk'][stimulus]
        epsilon = self.paras['epsilon']

        if not done:
            if reward == 1:
                # win stays
                pk = [epsilon/2] * 2
                pk[action] = 1 - epsilon/2
            else:
                pk = [1 - epsilon/2] * 2
                pk[action] = epsilon/2

        self.hidden_state['pk'][stimulus] = pk

        return pk

    def reset(self):
        self.hidden_state = self._set_hidden_state()
        return None

    def act(self, p):
        return np.random.choice(range(self.n_actions), p=p)