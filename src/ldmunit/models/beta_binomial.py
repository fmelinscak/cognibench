import sciunit
import numpy as np
from gym import spaces
from scipy.optimize import minimize
from scipy.stats import beta

class BetaBinomialModel(sciunit.Model):

    action_space = spaces.Box(-1000, 1000, shape=(1,), dtype=np.float32) #TODO: change
    observation_space = spaces.MultiBinary(2)

    def __init__(self, n_obs, paras=None, name=None):
        self.paras = paras
        self.name = name
        self.n_obs = n_obs
        self._set_spaces(n_obs)
        self.hidden_state = self._set_hidden_state(n_obs, self.paras)

    def _set_hidden_state(self, n_obs, paras):
        hidden_state = {'a' : np.zeros(n_obs),
                        'b' : np.zeros(n_obs)}
        return hidden_state

    def _set_spaces(self, n_obs):
        BetaBinomialModel.action_space = spaces.Box(-1000, 1000, shape=(1,), dtype=np.float32)
        BetaBinomialModel.observation_space = spaces.MultiBinary(n_obs)

    def predict(self, paras, stimulus):
        """Predict choice probabilities based on stimulus (observation in AI Gym)."""
        assert BetaBinomialModel.observation_space.contains(stimulus)

        # unpack parameters
        slope     = self.paras['slope']
        mixCoef   = self.paras['mixCoef']
        intercept = self.paras['intercept']

        # get model's state
        a = self.hidden_state['a']
        b = self.hidden_state['b']

        # Generate outcome prediction
        mu      = beta.mean(a, b)
        entropy = beta.entropy(a, b)

        return intercept + np.dot(stimulus, (mixCoef * mu  + (1 - mixCoef) * entropy)) * slope

    def update(self, stimulus, reward, action, done): #TODO: add default value
        """Update model's state given stimulus (observation in AI Gym), reward, action in the environment."""
        assert self.observation_space.contains(stimulus)
        # get model's state
        a = self.hidden_state['a']
        b = self.hidden_state['b']

        if not done:
            a += (1 - stimulus) * (1 - reward) + stimulus * reward
            b += (1 - stimulus) * reward + stimulus * (1 - reward)
            self.hidden_state['a'] = a
            self.hidden_state['b'] = b

        return a, b

    def reset(self):
        """Reset model's state."""
        self.hidden_state = self._set_hidden_state(self.n_obs, self.params)      
        return None
    
    def act(self):
        pass