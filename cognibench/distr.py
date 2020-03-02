import numpy as np
import scipy.stats as stats


class DiscreteRV:
    def __init__(self, p, eps=1e-8):
        self.n = len(p)
        self._p = p
        self._logp = np.log(p + eps)
        self.random_state = None

    def logpmf(self, e):
        return self._logp[e]

    def rvs(self):
        return self.random_state.choice(self.n, p=self._p)


#
#
# class ContinuousRV:
#    def __init__(self, rv_cont):
#        self._rv_cont = rv_cont
#
#    def __getitem__(self, e):
#        return
