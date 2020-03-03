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


class NormalRV:
    def __init__(self, loc, scale, eps=1e-8):
        self.loc = loc
        self.scale = scale
        self.eps = eps
        self.random_state = None

    def logpdf(self, e):
        return (
            -np.log(self.scale + self.eps)
            - 0.5 * np.log(2 * np.pi + self.eps)
            - 0.5 * ((e - self.loc) / (self.scale + self.eps)) ** 2
        )

    def rvs(self):
        return self.random_state.normal(self.loc, self.scale)
