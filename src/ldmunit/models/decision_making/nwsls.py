import sciunit
import numpy as np
from gym import spaces
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
        self.n_obs = n_obs
        self._set_spaces()
        self.hidden_state = self._set_hidden_state()

    def _set_hidden_state(self):
        xk = np.arange(self.n_actions)
        pk = np.full(self.n_actions, 1 / self.n_actions)
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

        rv = self.hidden_state['rv'][stimulus]
        epsilon = self.paras['epsilon']
        n = self.n_actions

        if not done:
            if reward == 1:
                # win stays
                prob_action = 1 - epsilon / n
                prob_others = (1 - prob_action) / (n - 1)
                pk = np.full(n, prob_others)
                pk[action] = prob_action
            else:
                # lose shift
                prob_action = epsilon / n
                prob_others = (1 - prob_action) / (n - 1)
                pk = np.full(n, prob_others)
                pk[action] = prob_action

        # update probability for this stimulus
        rv.pk = pk

        return pk

    def reset(self):
        self.hidden_state = self._set_hidden_state()
        return None

    def act(self, stimulus):
        assert self.observation_space.contains(stimulus)
        rv = self.hidden_state['rv'][stimulus]
        return rv.rvs()

