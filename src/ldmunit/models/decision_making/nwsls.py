import sciunit
import numpy as np
from gym import spaces
from scipy import stats

from .base import DADO
from ...capabilities import Interactive, LogProbModel

class NWSLSModel(DADO, Interactive, LogProbModel):
    """Noisy-win-stay-lose-shift model"""
    name = 'NWSLSModel'

    def __init__(self, n_action=None, n_obs=None, paras=None, hidden_state=None, seed=None, name=None, **params):
        return super().__init__(n_action=n_action, n_obs=n_obs, paras=paras, hidden_state=hidden_state, seed=seed, name=name, **params)
        
    def reset(self, paras=None):
        if not (isinstance(self.n_action, int) and isinstance(self.n_obs, int)):
            raise TypeError("action_space and observation_space must be set.")

        hidden_state = dict(win=True, action=np.random.randint(0,self.n_action))

        self.hidden_state = hidden_state

    def _get_rv(self, stimulus, paras=None):
        assert self.observation_space.contains(stimulus)

        if not paras:
            paras = self.paras
        
        epsilon = paras['epsilon']
        n = self.n_action

        if self.hidden_state['win']:
            prob_action = 1 - epsilon / n
        else:
            prob_action = epsilon / n
    
        pk = np.full(n, (1 - prob_action) / (n - 1))
        pk[self.hidden_state['action']] = prob_action

        xk = np.arange(n)
        rv = stats.rv_discrete(name=None, values=(xk, pk))
        if self.seed:
            rv.random_state = self.seed

        return rv

    def predict(self, stimulus, paras=None):
        return self._get_rv(stimulus, paras=paras).logpmf

    def act(self, stimulus, paras=None):
        return self._get_rv(stimulus, paras=paras).rvs()
        
    def update(self, stimulus, reward, action, done, paras=None):
        assert self.action_space.contains(action)
        assert self.observation_space.contains(stimulus)

        self.hidden_state['win'] = reward == 1
        self.hidden_state['action'] = action
