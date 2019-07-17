import numpy as np
from gym import spaces
from scipy import stats
from scipy.special import softmax

from .base import DADO
from ...capabilities import Interactive, LogProbModel

class RWCKModel(DADO, Interactive, LogProbModel):
    """
    Rescorla-Wagner Choice Kernel model implementation.

    Random variable for a given stimulus i is computed using ith row of Q
    matrix (Q_i), ith row of CK matrix (CK_i) and weights:

    >>> logits = beta * Q_i + beta_c * CK_i
    >>> probs = softmax(logits)
    """
    name = "RWCKModel"

    def __init__(self, *args, w, beta, beta_c, eta, eta_c, **kwargs):
        """
        Parameters
        ----------
        w : float
            Initial value of every element of weight matrix Q.

        beta : float
            Multiplicative factor used to multiply a row of Q matrix when computing
            logits.

        beta_c : float
            Multiplicative factor used to multiply a row of CK matrix when computing
            logits.

        eta : float
            Learning rate for Q updates. Must be nonnegative.

        eta_c : float
            Learning rate for CK updates. Must be nonnegative.
        """
        assert eta >= 0, 'eta must be nonnegative'
        assert eta_c >= 0, 'eta_c must be nonnegative'
        paras = {
            'w' : w,
            'beta' : beta,
            'beta_c' : beta_c,
            'eta' : eta,
            'eta_c' : eta_c
        }
        super().__init__(paras=paras, **kwargs)

    def reset(self):
        w = self.paras['w']
        self.hidden_state = {'CK': np.zeros((self.n_obs, self.n_action)),
                             'Q' : np.full((self.n_obs, self.n_action), w)}

    def _get_rv(self, stimulus):
        assert self.observation_space.contains(stimulus)
        CK_i = self.hidden_state['CK'][stimulus]
        Q_i = self.hidden_state['Q'][stimulus]

        beta   = self.paras['beta']
        beta_c = self.paras['beta_c']
        V = beta * Q_i + beta_c * CK_i

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
            eta   = self.paras['eta']
            eta_c = self.paras['eta_c']

            # update choice kernel
            CK = (1 - eta_c) * CK
            CK[action] += eta_c * reward

            # update Q weights
            delta = reward - Q[action]
            Q[action] += eta * delta

            self.hidden_state['CK'][stimulus] = CK
            self.hidden_state['Q' ][stimulus] = Q

        return CK, Q


class RWModel(RWCKModel):
    """
    Rescorla-Wagner model implementation as a special case of
    Rescorla-Wagner Choice Kernel model
    """
    name = "RWModel"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.paras['beta_c'] = 0


class CKModel(RWCKModel):
    """
    Choice Kernel model implementation as a special case of
    Rescorla-Wagner Choice Kernel model
    """
    name = "CKModel"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.paras['beta'] = 0
