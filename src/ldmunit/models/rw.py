import sciunit
import numpy as np
from ..capabilities import ProducesLoglikelihood
from ..utils import softmax
from numba import jit

class RWModel(sciunit.Model, ProducesLoglikelihood):
    """A decision making models (Rescorla–Wagner model)"""

    def __init__(self, alpha, beta, name=None):
        self.alpha = alpha
        self.beta = beta
        super(RWModel, self).__init__(name=name,
                                          alpha=alpha, beta=beta)
 
    def produce_loglikelihood(self, rewards, stimuli):

        def logpdf(actions): 
            """Calculate the negative log-likelihood score given actions."""
            prob_log = 0
            stimuli_list = np.unique(stimuli).tolist()
            n_actions = np.unique(actions).shape[0]

            Q = dict([[stimulus, np.zeros(n_actions)] for stimulus in stimuli_list])
            for action, reward, stimulus in zip(actions, rewards, stimuli):
                Q[stimulus][action] += self.alpha * (reward - Q[stimulus][action])
                prob_log += np.log(softmax(Q[stimulus], self.beta)[action])

            return -prob_log

        return logpdf
