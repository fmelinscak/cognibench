import sciunit
import numpy as np
from .. import capabilities
from .. import utils


def typecheck(types):
    def __f(f):
        def _f(*args, **kwargs):
            for a, t in zip(args, types):
                if not isinstance(a, t):
                    raise ValueError("Expected %s got %r" % (t, a))
            return f(*args, **kwargs)
        return _f
    return __f

class RWModel(sciunit.Model, capabilities.ProducesLoglikelihood):
    """"""

    def __init__(self, para, name=None):
        self.para = para
        super(RWModel, self).__init__(name=name, para=para)
    
    def produce_loglikelihood(self):
        return dict(zip(['para', 'lpdf'], 
        [self.para, utils.softmax]))
