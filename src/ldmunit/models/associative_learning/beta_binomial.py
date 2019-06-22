import sciunit
import numpy as np
from gym import spaces
from scipy.optimize import minimize
from scipy.stats import beta
from ..capabilities import Interactive

class BetaBinomialModel(sciunit.Model, Interactive):

    action_space = spaces.Box
    observation_space = spaces.MultiBinary

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

    def predict(self, stimulus):
        assert BetaBinomialModel.observation_space.contains(stimulus)
        # get model's state
        a = self.hidden_state['a']
        b = self.hidden_state['b']

        logpdf = lambda x: beta.logpdf(x, a=a, b=b)
        return logpdf

    def update(self, stimulus, reward, action, done): #TODO: add default value
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
        self.hidden_state = self._set_hidden_state(self.n_obs, self.params)      
        return None
    
    def act(self, stimulus):
        """observation function"""
        mixCoef = self.paras['mixCoef']
        b0 = self.paras['b0'] # intercept
        b1 = self.paras['b1'] # slope #TODO: add variable slopes

        # get model's state
        a = self.hidden_state['a']
        b = self.hidden_state['b']

        # Generate outcome prediction
        mu      = beta.mean(a, b)
        entropy = beta.entropy(a, b)
        
        return b0 + np.dot(stimulus, (mixCoef * mu  + (1 - mixCoef) * entropy)) * b1