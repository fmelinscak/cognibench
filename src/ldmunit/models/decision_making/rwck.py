import sciunit
import numpy as np
from gym import spaces
from scipy import stats
from scipy.special import softmax

from .base import DADO

class RWCKModel(DADO):
    """Rescorla Wagner Choice kernel Model for discrete decision marking."""
    name = "RWCKModel"

    def __init__(self, n_action=None, n_obs=None, paras=None, hidden_state=None, seed=None, name=None, **params):
        return super().__init__(n_action=n_action, n_obs=n_obs, paras=paras, hidden_state=hidden_state, seed=seed, name=name, **params)

    def reset(self, paras=None):
        if not (isinstance(self.n_action, int) and isinstance(self.n_obs, int)):
            raise TypeError("action_space and observation_space must be set.")
        if not paras:
            paras = self.paras

        w0 = paras['w0']

        hidden_state = {'CK': dict([[i, np.zeros(self.n_action)]    for i in range(self.n_obs)]),
                        'Q' : dict([[i, np.full(self.n_action, w0)] for i in range(self.n_obs)])}
        self.hidden_state = hidden_state

    def _get_rv(self, stimulus, paras=None):
        assert self.observation_space.contains(stimulus)
        CK, Q = self.hidden_state['CK'][stimulus], self.hidden_state['Q'][stimulus]

        if not paras:
            paras = self.paras

        beta   = paras['beta']
        beta_c = paras['beta_c']
        V = beta * Q + beta_c * CK

        xk = np.arange(self.n_action)
        pk = softmax(V)
        rv = stats.rv_discrete(values=(xk, pk))
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
        if not paras:
            paras = self.paras

        # get model's state
        CK, Q = self.hidden_state['CK'][stimulus], self.hidden_state['Q'][stimulus]
        
        if not done:
            # unpack parameters
            alpha   = paras['alpha'  ]
            alpha_c = paras['alpha_c']

            # update choice kernel
            CK = (1 - alpha_c) * CK
            CK[action] += alpha_c * reward

            # update Q weights
            delta = reward - Q[action]
            Q[action] += alpha * delta

            self.hidden_state['CK'][stimulus] = CK
            self.hidden_state['Q' ][stimulus] = Q

        return CK, Q

class RWModel(RWCKModel):
    """Rescorla Wagner Model for discrete decision marking."""
    name = "RWModel"
    
    def __init__(self, n_action=None, n_obs=None, paras=None, hidden_state=None, seed=None, name=None, **params):
        self.paras.update({'beta_c': 0})
        return super().__init__(n_action=n_action, n_obs=n_obs, paras=paras, hidden_state=hidden_state, seed=seed, name=name, **params)

class CKModel(RWCKModel):
    """Choice kernel Model for discrete decision marking."""
    name = "CKModel"

    def __init__(self, n_action=None, n_obs=None, paras=None, hidden_state=None, seed=None, name=None, **params):
        self.paras.update({'beta': 0})
        return super().__init__(n_action=n_action, n_obs=n_obs, paras=paras, hidden_state=hidden_state, seed=seed, name=name, **params)

