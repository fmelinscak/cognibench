import sciunit
from sciunit.capabilities import ProducesNumber
import numpy as np
from numpy.random import poisson
from .. import capabilities

"""SciUnit model classes."""

class PoissonModel(sciunit.Model, ProducesNumber):
    """A model that always produces a random number distributed in Poisson 
    distribution as output, with lambda = lam."""

    def __init__(self, lam, name=None):
        self.lam = lam
        super(PoissonModel, self).__init__(name=name, lam=lam)

    def produce_number(self):
        return poisson(self.lam)

class PoissonAltModel(sciunit.Model, capabilities.ProducesLoglikelihood):
    """A alternative way to represent Poisson model."""

    def __init__(self, para, name=None):
        self.para = para
        super(PoissonAltModel, self).__init__(name=name, para=para)

    def produce_loglikelihood(self):
        return dict(zip(['para', 'lpdf'], 
        # pdf of poi(para): e^-para * para^x / (x!)
        [self.para,  lambda para, obs: -para + obs * np.log(para)])) # terms w/ only observation was omitted
