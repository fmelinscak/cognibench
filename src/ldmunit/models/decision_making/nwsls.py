import sciunit
import numpy as np
from gym import spaces
from scipy import stats

from .base import DADO

class NWSLSModel(DADO):
    """Noisy-win-stay-lose-shift model"""

    def __init__(self, n_action=None, n_obs=None, paras=None, hidden_state=None, name=None, **params):
        return super().__init__(n_action=n_action, n_obs=n_obs, paras=paras, hidden_state=hidden_state, name=name, **params)

    def _get_default_paras(self):
        return {'epsilon': 0.5}

    def reset(self):
        if not (isinstance(self.n_action, int) and isinstance(self.n_obs, int)):
            raise TypeError("action_space and observation_space must be set.")

        xk = np.arange(self.n_action)
        pk = np.full(self.n_action, 1 / self.n_action)
        rv = stats.rv_discrete(values=(xk, pk))

        hidden_state = {'rv': dict([[i, rv] for i in range(self.n_obs)])}
        self.hidden_state = hidden_state

    def predict(self, stimulus):
        assert self.observation_space.contains(stimulus)

        rv = self.hidden_state['rv'][stimulus]
        return rv.logpmf

    def update(self, stimulus, reward, action, done):
        assert self.action_space.contains(action)
        assert self.observation_space.contains(stimulus)

        rv = self.hidden_state['rv'][stimulus]
        epsilon = self.paras['epsilon']
        n = self.n_action

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

    def act(self, stimulus):
        assert self.observation_space.contains(stimulus)
        rv = self.hidden_state['rv'][stimulus]
        return rv.rvs()

