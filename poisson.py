import sciunit
from sciunit.capabilities import ProducesNumber
import numpy as np
from numpy.random import poisson

from .capabilities import ProducesLoglikelihood
from .utils import softmax

"""SciUnit model classes."""

class PoissonModel(sciunit.Model, ProducesNumber):
    """A model that always produces a random number distributed in Poisson 
    distribution as output, with lambda = lam."""

    def __init__(self, lam, name=None):
        self.lam = lam
        super(PoissonModel, self).__init__(name=name, lam=lam)

    def produce_number(self):
        return poisson(self.lam)
