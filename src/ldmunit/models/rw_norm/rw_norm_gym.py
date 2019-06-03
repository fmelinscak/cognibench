import sciunit
import numpy as np
import gym
from gym import spaces
from scipy.optimize import minimize
from ...capabilities import SupportContinuousActions

class RwNormGymModel(sciunit.Model):
    def __init__(self, n_obs, paras=None, name=None):
        assert isinstance(n_obs, int)
        self.paras = paras
        self.name = name
        self.n_obs = n_obs
        print("#TODO: change the action_space")
        self.action_space = spaces.Box(-1000, 1000, shape=(1,), dtype=np.float32) #TODO:
        self.observation_space = spaces.MultiBinary(self.n_obs)

        # init model's state
        self.hidden_state = {'w'    : np.zeros(n_obs),
                             'rhat' : np.random.ranf(1)} #TODO: store reward predict


    def predict(self, paras, stimulus):
        """Predict choice probabilities based on stimulus (observation in AI Gym)."""
        assert self.observation_space.contains(stimulus)

        # unpack parameters
        b0      = paras['b0']
        b1      = paras['b1']
        sd_pred = paras['sigma']

        # get model's weight
        w_curr = self.hidden_state['w']

        # Generate outcome prediction
        rhat = np.dot(stimulus, w_curr.T)

        # store rhat
        self.hidden_state['rhat'] = rhat
        
        # Predict response
        mu_pred = b0 + b1 * rhat

        return mu_pred, sd_pred

    def update(self, paras, stimulus, reward, action, done): #TODO: add default value
        """Update model's state given stimulus (observation in AI Gym), reward, action in the environment."""
        assert self.observation_space.contains(stimulus)
        # get model's state
        w_curr = self.hidden_state['w']

        if not done:
            # unpack parameters
            alpha = paras['alpha']

            # Calculate prediction error
            pred_err = reward - action

            # update weights
            w_curr += alpha * pred_err * stimulus
            self.hidden_state['w'] = w_curr

        return w_curr

    def reset(self):
        """Reset model's state."""
        self.hidden_state = {'w'    : np.zeros(self.n_obs),
                             'rhat' : np.random.ranf(1)}
        
        return 0 #TODO
    
    def act(self):
        """Agent does not exert action in the environment"""
        return self.hidden_state['rhat']