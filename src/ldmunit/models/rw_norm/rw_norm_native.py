import sciunit
import numpy as np
import gym
from gym import spaces
from scipy.optimize import minimize

class RwNormGymModel(sciunit.Model):
    
    action_space = spaces.Box(-1000, 1000, shape=(1,), dtype=np.float32) #TODO: change
    observation_space = spaces.MultiBinary(2)

    def __init__(self, n_obs, paras=None, name=None):
        assert isinstance(n_obs, int)
        self.paras = paras
        self.name = name
        self.n_obs = n_obs
        self._set_spaces(n_obs)
        self.hidden_state = self._set_hidden_state(n_obs, self.paras)

    def _set_hidden_state(self, n_obs, paras):
        hidden_state = {'w'    : np.zeros(n_obs),
                        'rhat' : np.random.ranf(1)}
        return hidden_state

    def _set_spaces(self, n_obs):
        RwNormGymModel.action_space = spaces.Box(-1000, 1000, shape=(1,), dtype=np.float32)
        RwNormGymModel.observation_space = spaces.MultiBinary(n_obs)

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
        self.hidden_state = self._set_hidden_state(self.n_obs, self.params)
        return None
    
    def act(self):
        pass