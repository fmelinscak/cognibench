import sciunit
import numpy as np
from gym import spaces
from scipy import stats

from .base import DADO

def softmax(x, beta):
    return np.exp(x * beta) / np.sum(np.exp(x * beta), axis=0)

class RWCKModel(DADO):
    """Rescorla Wagner Choice kernel Model for discrete decision marking."""

    def __init__(self, n_action=None, n_obs=None, paras=None, hidden_state=None, name=None, **params):
        return super().__init__(n_action=n_action, n_obs=n_obs, paras=paras, hidden_state=hidden_state, name=name, **params)

    def _get_default_paras(self):
        return {'beta': 0.5, 'beta_c': 1, 'alpha': 1, 'alpha_c': 1, 'w0': 0}

    def reset(self):
        if not (isinstance(self.n_action, int) and isinstance(self.n_obs, int)):
            raise TypeError("action_space and observation_space must be set.")

        w0 = self.paras['w0']  

        # set rv_discrete for each stimulus/cue/observation
        xk = np.arange(self.n_action)
        pk = np.full(self.n_action, 1 / self.n_action)
        rv = stats.rv_discrete(name=self.name, values=(xk, pk))

        hidden_state = {'CK': dict([[i, np.zeros(self.n_action)]    for i in range(self.n_obs)]),
                        'Q' : dict([[i, np.full(self.n_action, w0)] for i in range(self.n_obs)]),
                        'rv': dict([[i, rv]                         for i in range(self.n_obs)])}
        self.hidden_state = hidden_state

    def predict(self, stimulus):
        assert self.observation_space.contains(stimulus)
        # get model's state
        CK, Q = self.hidden_state['CK'][stimulus], self.hidden_state['Q'][stimulus]
        rv = self.hidden_state['rv'][stimulus]
        
        # unpack parameters
        beta   = self.paras['beta']
        beta_c = self.paras['beta_c' ]

        V = beta * Q + beta_c * CK
        pk = softmax(V, 1)

        # update pk
        rv.pk = pk
        
        return rv.logpmf

    def update(self, stimulus, reward, action, done): #TODO: add default value
        assert self.action_space.contains(action)
        assert self.observation_space.contains(stimulus)

        # get model's state
        CK, Q = self.hidden_state['CK'][stimulus], self.hidden_state['Q'][stimulus]
        
        if not done:
            # unpack parameters
            alpha   = self.paras['alpha'  ]
            alpha_c = self.paras['alpha_c']

            # update choice kernel
            CK = (1 - alpha_c) * CK
            CK[action] += alpha_c * reward

            # update Q weights
            delta = reward - Q[action]
            Q[action] += alpha * delta

            self.hidden_state['CK'][stimulus] = CK
            self.hidden_state['Q' ][stimulus] = Q

        return CK, Q

    def act(self, stimulus):
        rv = self.hidden_state['rv'][stimulus]
        return rv.rvs()

class RWModel(RWCKModel):
    """Rescorla Wagner Model for discrete decision marking."""

    def __init__(self, n_action=None, n_obs=None, paras=None, hidden_state=None, name=None, **params):
        self.paras.update({'beta_c': 0})
        return super().__init__(n_action=n_action, n_obs=n_obs, paras=paras, hidden_state=hidden_state, name=name, **params)

class CKModel(RWCKModel):
    """Choice kernel Model for discrete decision marking."""

    def __init__(self, n_action=None, n_obs=None, paras=None, hidden_state=None, name=None, **params):
        self.paras.update({'beta': 0})
        return super().__init__(n_action=n_action, n_obs=n_obs, paras=paras, hidden_state=hidden_state, name=name, **params)

