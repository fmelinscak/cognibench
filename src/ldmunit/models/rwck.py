import sciunit
import numpy as np
from gym import spaces
from scipy.optimize import minimize

def softmax(x, beta):
    return np.exp(x * beta) / np.sum(np.exp(x * beta), axis=0)

class RWCKModel(sciunit.Model):

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
        w0 = 0
        if 'w0' in paras:
            w0 = paras['w0']

        hidden_state = {'CK': dict([[i, np.zeros(n_actions)]    for i in range(n_obs)]),
                        'Q' : dict([[i, np.full(n_actions, w0)] for i in range(n_obs)])}

        return hidden_state

    def _set_spaces(self, n_actions):
        RWCKModel.action_space = spaces.Discrete(n_actions)
        RWCKModel.observation_space = spaces.Discrete(n_actions)

    def predict(self, stimulus):
        """Predict choice probabilities based on stimulus (observation in AI Gym)."""
        assert RWCKModel.observation_space.contains(stimulus)
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
        assert RWCKModel.action_space.contains(action)
        assert RWCKModel.observation_space.contains(stimulus)
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
        self.hidden_state = self._set_hidden_state(self.n_actions, self.n_obs, self.paras)
        return None
    
    def act(self, p):
        """Agent make decision/choice based on the probabilities."""
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

class RWModel(RWCKModel):

    def __init__(self, n_actions, n_obs, paras=None, name=None):
        self.paras = paras
        self.paras.update({'beta_c': 0})
        self.name = name
        self.n_actions = n_actions
        self.n_obs = n_obs
        self._set_spaces(n_actions)
        self.hidden_state = self._set_hidden_state(n_actions, n_obs, self.paras)

class CKModel(RWCKModel):

    def __init__(self, n_actions, n_obs, paras=None, name=None):
        self.paras = paras
        self.paras.update({'beta': 0})
        self.name = name
        self.n_actions = n_actions
        self.n_obs = n_obs
        self._set_spaces(n_actions)
        self.hidden_state = self._set_hidden_state(n_actions, n_obs, self.paras)