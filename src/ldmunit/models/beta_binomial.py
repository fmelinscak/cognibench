import sciunit
import numpy as np
from gym import spaces
from scipy.optimize import minimize
from scipy.stats import beta

class NWSLSModel(sciunit.Model):

    action_space = spaces.Discrete(1)
    observation_space = spaces.Discrete(1)

    """Noisy-win-stay-lose-shift model"""
    def __init__(self, n_actions, n_obs, paras=None, name=None):
        self.paras = paras
        self.name = name
        self.n_actions = n_actions
        self.n_obs = n_obs # number of stimuli
        self._set_spaces(n_actions)
        self.hidden_state = self._set_hidden_state(n_actions, n_obs, self.paras)

    def _set_hidden_state(self, n_actions, n_obs, paras):
        hidden_state = {'P': dict([[i, np.full(n_actions, 1/n_actions)] for i in range(n_obs)])}
        return hidden_state

    def _set_spaces(self, n_actions):
        NWSLSModel.action_space = spaces.Discrete(n_actions)
        NWSLSModel.observation_space = spaces.Discrete(n_actions)

    def predict(self, stimulus):
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
        self.hidden_state = self._set_hidden_state(self.n_actions, self.n_obs, self.paras)
        return None

    def act(self, p):
        return np.random.choice(range(self.n_actions), p=p)

    def loglike(self, stimuli, rewards, actions):
        #TODO: add assertion for the length
        n_trials = len(stimuli)
        
        res = 0
        
        hidden_state = self.hidden_state
        
        for i in range(n_trials):
            # compute choice probabilities
            P = self.predict(stimuli[i])
            
            # probability of the action
            #TODO: generalize this
            p = P[actions[i]]

            # add log-likelihood
            res += np.log(p)
            # update choice kernel and Q weights
            self.update(stimuli[i], rewards[i], actions[i], False)
        
        # keep the model's hidden state intact
        self.hidden_state = hidden_state
        
        return res

    def train_with_obs(self, stimuli, rewards, actions, fixed):

        x0 = list(fixed.values())

        def objective_func(x0):
            for k, v in zip(fixed.keys(), x0):
                self.paras[k] = v
            return - self.loglike(stimuli, rewards, actions)

        opt_results = minimize(fun=objective_func, x0=x0) 

        return opt_results
