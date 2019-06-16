import sciunit
import numpy as np
from gym import spaces
from scipy.optimize import minimize

class RRModel(sciunit.Model):

    action_space = spaces.Discrete(1)
    observation_space = spaces.Discrete(1)

    """Rescorla Wagner Choice kernel Model for discrete decision marking."""
    def __init__(self, n_actions, n_obs, paras=None, name=None):
        self.paras = paras
        self.name = name
        self.n_actions = n_actions
        self.n_obs = n_obs
        self._set_spaces(n_actions)
        self.hidden_state = self._set_hidden_state(n_actions, n_obs, self.paras)
        
    def _set_hidden_state(self, n_actions, n_obs, paras):
        hidden_state = None
        return hidden_state

    def _set_spaces(self, n_actions):
        RRModel.action_space = spaces.Discrete(n_actions)
        RRModel.observation_space = spaces.Discrete(n_actions)

    def predict(self, stimulus):
        """Predict choice probabilities based on stimulus (observation in AI Gym)."""
        pass

    def update(self, stimulus, reward, action, done): #TODO: add default value
        """Update model's state given stimulus (observation in AI Gym), reward, action in the environment."""
        assert RRModel.action_space.contains(action)
        assert RRModel.observation_space.contains(stimulus)
        pass

    def reset(self):
        """Reset model's state."""
        self.hidden_state = self._set_hidden_state(self.n_actions, self.n_obs, self.paras)
        return None
    
    def act(self, p):
        """Agent make decision/choice based on the probabilities."""
        pass

    def loglike(self, stimuli, rewards, actions):
        pass

    def train_with_obs(self, stimuli, rewards, actions, fixed):
        pass
