import sciunit
import numpy as np
from gym import spaces
from scipy import stats

from ...capabilities import Interactive, DiscreteAction, DiscreteObservation

def softmax(x, beta):
    return np.exp(x * beta) / np.sum(np.exp(x * beta), axis=0)

class RWCKModel(sciunit.Model, Interactive, DiscreteAction, DiscreteObservation):
    """Rescorla Wagner Choice kernel Model for discrete decision marking."""

    action_space = spaces.Discrete
    observation_space = spaces.Discrete

    def __init__(self, n_actions, n_obs, paras=None, name=None):
        self.paras = paras
        self.name = name
        self.n_actions = n_actions
        self.n_obs = n_obs
        self.hidden_state = self._set_hidden_state()
        self.action_space = self._set_action_space()
        self.observation_space = self._set_observation_space()

        
    def _set_hidden_state(self):
        w0 = 0 if 'w0' not in self.paras else self.paras['w0']  

        # set rv_discrete for each stimulus/cue/observation
        xk = np.arange(self.n_actions)
        pk = np.full(self.n_actions, 1 / self.n_actions)
        rv = stats.rv_discrete(name=self.name, values=(xk, pk))

        hidden_state = {'CK': dict([[i, np.zeros(self.n_actions)]    for i in range(self.n_obs)]),
                        'Q' : dict([[i, np.full(self.n_actions, w0)] for i in range(self.n_obs)]),
                        'rv': dict([[i, rv]                          for i in range(self.n_obs)])}

        return hidden_state

    def _set_action_space(self):
        return spaces.Discrete(self.n_actions)
    
    def _set_observation_space(self):
        return spaces.Discrete(self.n_obs)

    def reset(self):
        self.action_space = self._set_action_space()
        self.observation_space = self._set_observation_space()

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

    def __init__(self, n_actions, n_obs, paras=None, name=None):
        self.paras = paras
        self.paras.update({'beta_c': 0})
        self.name = name
        self.n_actions = n_actions
        self.n_obs = n_obs
        self.hidden_state = self._set_hidden_state()
        self.action_space = self._set_action_space()
        self.observation_space = self._set_observation_space()

class CKModel(RWCKModel):
    """Choice kernel Model for discrete decision marking."""

    def __init__(self, n_actions, n_obs, paras=None, name=None):
        self.paras = paras
        self.paras.update({'beta': 0})
        self.name = name
        self.n_actions = n_actions
        self.n_obs = n_obs
        self.hidden_state = self._set_hidden_state()
        self.action_space = self._set_action_space()
        self.observation_space = self._set_observation_space()

