import sciunit
import numpy as np
import gym
from gym import spaces
from scipy.optimize import minimize
from ..capabilities import SupportsDiscreteActions

class NWSLSModel(sciunit.Model, SupportsDiscreteActions):
    """Noisy-win-stay-lose-shift model"""
    def __init__(self, n_actions, n_obs, paras=None, name=None):
        assert n_actions == 2 # two actions only
        assert isinstance(n_obs, int)
        self.paras = paras
        self.name = name
        self.n_actions = n_actions
        self.n_obs = n_obs # number of stimuli
        self.action_space = spaces.Discrete(self.n_actions)
        self.observation_space = spaces.Discrete(self.n_obs)
        # init model's state: last reward
        self.hidden_state = {'P': dict([[i, np.full(n_actions, 1/n_actions)] for i in range(n_obs)])}

    def predict(self, stimulus):
        """Predict choice probabilities based on stimulus (observation in AI Gym)."""
        assert self.observation_space.contains(stimulus)
        assert self.paras != None #TODO: add assert for keys
        # list of prob. for each subjects
        return self.hidden_state['P'][stimulus]

    def update(self, stimulus, reward, action, done):
        P = self.hidden_state['P'][stimulus]
        # unpack parameters
        epsilon = self.paras['epsilon']

        if not done:
            if reward == 1:
                # win stays
                P = [epsilon/2] * 2
                P[action] = 1 - epsilon/2
            else:
                P = [1 - epsilon/2] * 2
                P[action] = epsilon/2

        self.hidden_state['P'][stimulus] = P

        return P

    def reset(self):
        """Reset model's state."""
        self.hidden_state = {'P': dict([[i, np.full(self.n_actions, 1/self.n_actions)] for i in range(self.n_obs)])}

        return self.action_space.sample()
    
    def act(self, p):
        """Agent make decision/choice based on the probabilities."""
        assert len(p) == self.n_actions
        return np.random.choice(range(self.n_actions), p=p)

    def loglikelihood(self, P, action):
        """Return log-likelihood value of action based on P given by predict() method"""
        return P[action]

class NWSLSMultiModel(sciunit.Model, SupportsDiscreteActions):
    """Noisy-win-stay-lose-shift model"""
    def __init__(self, n_actions, n_obs, paras=None, name=None):
        assert n_actions == 2 # two actions only
        assert isinstance(n_obs, int)
        self.paras = paras
        self.name = name
        self.n_actions = n_actions
        self.n_obs = n_obs # number of stimuli
        self.action_space = spaces.Discrete(self.n_actions)
        self.observation_space = spaces.Discrete(self.n_obs)
        # init model's state: last reward
        self.n_sub = len(self.paras)

        hidden_state = {'P': dict([[i, np.full(n_actions, 1/n_actions)] for i in range(n_obs)])}

        self.hidden_state = [hidden_state] * self.n_sub


    def predict(self, stimulus, sub):
        """Predict choice probabilities based on stimulus (observation in AI Gym)."""
        assert self.observation_space.contains(stimulus)
        assert self.paras != None #TODO: add assert for keys
        # list of prob. for each subjects
        return self.hidden_state[sub]['P'][stimulus]

    def update(self, stimulus, reward, action, done, sub):
        P = self.hidden_state[sub]['P'][stimulus]
        # unpack parameters
        epsilon = self.paras[sub]['epsilon']

        if not done:
            if reward == 1:
                # win stays
                P = [epsilon/2] * 2
                P[action] = 1 - epsilon/2
            else:
                P = [1 - epsilon/2] * 2
                P[action] = epsilon/2

        self.hidden_state[sub]['P'][stimulus] = P

        return P

    def reset(self):
        """Reset model's state."""
        hidden_state = {'P': dict([[i, np.full(self.n_actions, 1/self.n_actions)] for i in range(self.n_obs)])}

        self.hidden_state = [hidden_state] * self.n_sub
        return self.action_space.sample()
    
    def act(self, p):
        """Agent make decision/choice based on the probabilities."""
        assert len(p) == self.n_actions
        return np.random.choice(range(self.n_actions), p=p)
