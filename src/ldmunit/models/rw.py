import sciunit
import numpy as np
from ..capabilities import ProducesLoglikelihood
from ..utils import softmax
from numba import jit

@jit("float64[:](float64[:], float64[:], float64)", nopython=False, nogil=True)
def calc_weight(rewards, actions, alpha):
    """Recursive calculate the weight in numpy array (float64)."""
    Q = np.zeros_like(actions, dtype="float64") # must specify the dtype
    for i in range(1, actions.shape[0]):
        Q[i,:] = Q[i-1,:] + alpha * actions[i-1,:] * (rewards[i-1] - Q[i-1,:])
        # Numerically equal to Q_{n+1}^k = Q_n^k + \alpha * (r_n - Q_n^k)
    return Q

class RWModel(sciunit.Model, ProducesLoglikelihood):
    """A decision making models (Rescorla–Wagner model)"""

    def __init__(self, alpha, beta, name=None):
        self.alpha = alpha
        self.beta = beta
        super(RWModel, self).__init__(name=name,
                                          alpha=alpha, beta=beta)
    
    def produce_loglikelihood(self, rewards):
        """Rewards and actions both needed for calculation."""

        def logpdf(actions): 
            """Calculate the negative log-likelihood score given actions."""
            Q = calc_weight(rewards, actions, self.alpha)
            # apply softmax along axis = 1
            prob = np.apply_along_axis(softmax, 1, Q, self.beta)

            # P(x|para.)
            prob_actions = actions * np.log(prob)
            return -np.sum(prob_actions)

        return logpdf
