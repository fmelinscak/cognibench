import numpy as np
import gym
from gym import spaces
from scipy import stats
from .base import CAMO
from ...capabilities import Interactive, LogProbModel

class LSSPDModel(CAMO, Interactive, LogProbModel):
    name = 'LSSPD'

    def __init__(self, *args, w0, alpha, b0, b1, sigma, mix_coef, eta, kappa, **kwargs):
        paras = {
            'w0' : w0,
            'alpha' : alpha,
            'b0' : b0,
            'b1' : b1,
            'sigma' : sigma,
            'mix_coef' : mix_coef,
            'eta' : eta,
            'kappa' : kappa
        }
        super().__init__(paras=paras, **kwargs)

    def reset(self):
        w0 = self.paras['w0'] if 'w0' in self.paras else 0
        alpha = self.paras['alpha'] if 'alpha' in self.paras else 0

        try:
            it = iter(w0)
            w0 = np.array(w0)
        except TypeError:
            w0 = np.full(self.n_obs, w0)

        try:
            it = iter(alpha)
            alpha = np.array(alpha)
        except TypeError:
            alpha = np.full(self.n_obs, alpha)

        self.hidden_state = {'w'    : w0,
                             'alpha': alpha}

    def observation(self, stimulus):
        assert self.observation_space.contains(stimulus)

        b0 = self.paras['b0'] # intercept
        b1 = self.paras['b1'] # slope
        sd_pred = self.paras['sigma']
        mix_coef = self.paras['mix_coef'] # proportion of the weights signal in the mixture of weight and associability signals
        
        w_curr = self.hidden_state['w']
        alpha  = self.hidden_state['alpha']

        # Predict response
        mu_pred = b0 + b1 * np.dot(stimulus, (mix_coef * w_curr + (1 - mix_coef) * alpha))

        rv = stats.norm(loc=mu_pred, scale=sd_pred)
        rv.random_state = self.seed

        return rv

    def predict(self, stimulus):
        return self.observation(stimulus).logpdf

    def act(self, stimulus):
        return self.observation(stimulus).rvs()

    def _predict_reward(self, stimulus):
        assert self.observation_space.contains(stimulus)
        w_curr = self.hidden_state['w']
        rhat = np.dot(stimulus, w_curr.T)
        return rhat
        
    def update(self, stimulus, reward, action, done):
        assert self.action_space.contains(action)
        assert self.observation_space.contains(stimulus)

        eta   = self.paras['eta'] # Proportion of pred. error. in the updated associability value
        kappa = self.paras['kappa'] # Fixed learning rate for the cue weight update
        
        w_curr = self.hidden_state['w']
        alpha  = self.hidden_state['alpha']

        rhat = self._predict_reward(stimulus)


        if not done:
            delta = reward - rhat

            w_curr += kappa * delta * alpha * stimulus # alpha, stimulus size: (n_obs,)

            # if stimulus[i] = 1
            # alpha[i] = eta * abs(pred_err) + (1 - eta) * alpha[i]
            # or
            # alpha[i] -= eta * alpha[i]
            # alpha[i] += eta * abs(pred_err)
            alpha -= eta * np.multiply(alpha, stimulus)
            alpha += eta * abs(delta) * stimulus
            np.clip(alpha, a_min=0, a_max=1, out=alpha) # Enforce upper bound on alpha

            self.hidden_state['w'] = w_curr
            self.hidden_state['alpha'] = alpha

        return w_curr, alpha
