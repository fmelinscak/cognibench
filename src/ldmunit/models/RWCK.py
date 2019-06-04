import sciunit
import numpy as np
import gym
from gym import spaces
from scipy.optimize import minimize
from ..capabilities import SupportsDiscreteActions

def softmax(x, beta):
    return np.exp(x * beta) / np.sum(np.exp(x * beta), axis=0)


class RWCKModel(sciunit.Model, SupportsDiscreteActions):
    """Rescorla Wagner Choice kernel Model for discrete decision marking."""
    def __init__(self, n_actions, n_obs, paras=None, name=None):
        assert isinstance(n_actions, int)
        assert isinstance(n_obs, int)
        self.paras = paras
        self.name = name
        self.n_actions = n_actions
        self.n_obs = n_obs
        self.action_space = spaces.Discrete(self.n_actions)
        self.observation_space = spaces.Discrete(self.n_obs)

        # init model's state #TODO: add init w0 in 'Q'
        self.hidden_state = {'CK': dict([[i, np.zeros(n_actions)] for i in range(n_obs)]),
                             'Q' : dict([[i, np.zeros(n_actions)] for i in range(n_obs)])}

    def predict(self, stimulus):
        """Predict choice probabilities based on stimulus (observation in AI Gym)."""
        assert self.observation_space.contains(stimulus)
        assert self.paras != None #TODO: add assert for keys
        # get model's state
        CK, Q = self.hidden_state['CK'][stimulus], self.hidden_state['Q'][stimulus]
        
        # unpack parameters
        beta   = self.paras['beta']
        beta_c = self.paras['beta_c' ]

        V = beta * Q + beta_c * CK
        P = softmax(V, 1)

        return P

    def update(self, stimulus, reward, action, done): #TODO: add default value
        """Update model's state given stimulus (observation in AI Gym), reward, action in the environment."""
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
        """Reset model's state."""
        self.hidden_state = {'CK': dict([[i, np.zeros(self.n_actions)] for i in range(self.n_obs)]),
                             'Q' : dict([[i, np.zeros(self.n_actions)] for i in range(self.n_obs)])}
        
        return self.action_space.sample()
    
    def act(self, p):
        """Agent make decision/choice based on the probabilities."""
        assert len(p) == self.n_actions
        return np.random.choice(range(self.n_actions), p=p)

    def loglikelihood(self, P, action):
        """Return log-likelihood value of action based on P given by predict() method"""
        return P[action]