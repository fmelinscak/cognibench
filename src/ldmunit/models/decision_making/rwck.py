import sciunit
import numpy as np
from gym import spaces
from scipy.optimize import minimize
from scipy import stats
from ...capabilities import Interactive

def softmax(x, beta):
    return np.exp(x * beta) / np.sum(np.exp(x * beta), axis=0)

class RWCKModel(sciunit.Model, Interactive):
    """Rescorla Wagner Choice kernel Model for discrete decision marking."""

    action_space = spaces.Discrete
    observation_space = spaces.Discrete

    def __init__(self, n_actions, n_obs, paras=None, name=None):
        self.paras = paras
        self.name = name
        self.n_actions = n_actions
        self.n_obs = n_obs
        self._set_spaces(n_actions)
        self.hidden_state = self._set_hidden_state(n_actions, n_obs, self.paras)

        
    def _set_hidden_state(self, n_actions, n_obs):
        w0 = 0
        if 'w0' in self.paras:
            w0 = self.paras['w0']

        # set rv_discrete for each stimulus/cue/observation
        xk = np.arange(n_actions)
        pk = np.full(n_actions, 1/n_actions)
        rv = stats.rv_discrete(name=self.name, values=(xk, pk))

        hidden_state = {'CK': dict([[i, np.zeros(n_actions)]    for i in range(n_obs)]),
                        'Q' : dict([[i, np.full(n_actions, w0)] for i in range(n_obs)]),
                        'RV': dict([[i, rv]                     for i in range(n_obs)])}

        return hidden_state

    def _set_spaces(self, n_actions):
        self.action_space = spaces.Discrete(n_actions)
        self.observation_space = spaces.Discrete(n_actions)

    def predict(self, stimulus):
        assert self.observation_space.contains(stimulus)
        assert self.paras != None #TODO: add assert for keys
        # get model's state
        CK, Q = self.hidden_state['CK'][stimulus], self.hidden_state['Q'][stimulus]
        RV = self.hidden_state['RV'][stimulus]
        
        # unpack parameters
        beta   = self.paras['beta']
        beta_c = self.paras['beta_c' ]

        V = beta * Q + beta_c * CK
        pk = softmax(V, 1)

        # update pk
        RV.pk = pk
        self.hidden_state['RV'][stimulus] = RV
        
        return RV.logpmf

    def update(self, stimulus, reward, action, done): #TODO: add default value
        assert self.action_space.contains(action)
        assert self.observation_space.contains(stimulus)
        assert self.paras != None #TODO: add assert for keys

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

    def reset(self):
        self.hidden_state = self._set_hidden_state(self.n_actions, self.n_obs)
        return None
    
    def act(self, stimulus):
        RV = self.hidden_state['RV'][stimulus]
        return RV.rvs()

class RWModel(RWCKModel):
    """Rescorla Wagner Model for discrete decision marking."""

    def __init__(self, n_actions, n_obs, paras=None, name=None):
        self.paras = paras
        self.paras.update({'beta_c': 0})
        self.name = name
        self.n_actions = n_actions
        self.n_obs = n_obs
        self._set_spaces(n_actions)
        self.hidden_state = self._set_hidden_state(n_actions, n_obs, self.paras)

class CKModel(RWCKModel):
    """Choice kernel Model for discrete decision marking."""

    def __init__(self, n_actions, n_obs, paras=None, name=None):
        self.paras = paras
        self.paras.update({'beta': 0})
        self.name = name
        self.n_actions = n_actions
        self.n_obs = n_obs
        self._set_spaces(n_actions)
        self.hidden_state = self._set_hidden_state(n_actions, n_obs, self.paras)