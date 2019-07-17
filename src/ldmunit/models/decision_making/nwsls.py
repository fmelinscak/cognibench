import numpy as np
from gym import spaces
from scipy import stats

from .base import DADO
from ...capabilities import Interactive, LogProbModel

class NWSLSModel(DADO, Interactive, LogProbModel):
    """
    Noisy-win-stay-lose-shift model implementation.
    """
    name = 'NWSLSModel'

    def __init__(self, *args, epsilon, **kwargs):
        """
        Parameters
        ----------
        epsilon : int
            Number of loose actions. Must be nonnegative and less than or equal
            to the dimension of the action space.
        """
        paras = dict(epsilon=epsilon)
        super().__init__(paras=paras, **kwargs)
        assert epsilon >= 0 and epsilon <= self.n_action, 'epsilon must be in range [0, n_action]'

    def reset(self):
        self.hidden_state = dict(win=True, action=self.rng.randint(0, self.n_action))

    def _get_rv(self, stimulus):
        assert self.observation_space.contains(stimulus)

        epsilon = self.paras['epsilon']
        n = self.n_action

        if self.hidden_state['win']:
            prob_action = 1 - epsilon / n
        else:
            prob_action = epsilon / n
    
        pk = np.full(n, (1 - prob_action) / (n - 1))
        pk[self.hidden_state['action']] = prob_action

        xk = np.arange(n)
        rv = stats.rv_discrete(name=None, values=(xk, pk))
        rv.random_state = self.seed

        return rv

    def predict(self, stimulus):
        return self._get_rv(stimulus).logpmf

    def act(self, stimulus):
        return self._get_rv(stimulus).rvs()
        
    def update(self, stimulus, reward, action, done):
        assert self.action_space.contains(action)
        assert self.observation_space.contains(stimulus)

        self.hidden_state['win'] = reward == 1
        self.hidden_state['action'] = action
