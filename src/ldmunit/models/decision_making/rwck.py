import numpy as np
from gym import spaces
from scipy import stats
from scipy.special import softmax

from .base import DADO
from ...capabilities import Interactive, LogProbModel

class RWCKModel(DADO, Interactive, LogProbModel):
    """Rescorla Wagner Choice kernel Model for discrete decision marking."""
    name = "RWCKModel"

    def __init__(self, *args, w0, beta, beta_c, alpha, alpha_c, **kwargs):
        paras = {
            'w0' : w0,
            'beta' : beta,
            'beta_c' : beta_c,
            'alpha' : alpha,
            'alpha_c' : alpha_c
        }
        super().__init__(paras=paras, **kwargs)

    def reset(self):
        w0 = self.paras['w0']
        self.hidden_state = {'CK': np.zeros((self.n_obs, self.n_action)),
                             'Q' : np.full((self.n_obs, self.n_action), w0)}

    def _get_rv(self, stimulus):
        assert self.observation_space.contains(stimulus)
        CK, Q = self.hidden_state['CK'][stimulus], self.hidden_state['Q'][stimulus]

        beta   = self.paras['beta']
        beta_c = self.paras['beta_c']
        V = beta * Q + beta_c * CK

        xk = np.arange(self.n_action)
        pk = softmax(V)
        rv = stats.rv_discrete(values=(xk, pk))
        rv.random_state = self.seed

        return rv

    def predict(self, stimulus):
        return self._get_rv(stimulus).logpmf

    def act(self, stimulus):
        return self._get_rv(stimulus).rvs()

    def update(self, stimulus, reward, action, done):
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


class RWModel(RWCKModel):
    """Rescorla Wagner Model for discrete decision marking."""
    name = "RWModel"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.paras['beta_c'] = 0


class CKModel(RWCKModel):
    """Choice kernel Model for discrete decision marking."""
    name = "CKModel"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.paras['beta'] = 0
