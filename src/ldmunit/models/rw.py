import sciunit
import numpy as np
from ..capabilities import ProducesLoglikelihood

template="""
import numpy as np
def softmax(x, beta):
    return np.exp(x * beta) / np.sum(np.exp(x * beta), axis=0)
def func(variable, stimuli, rewards, actions, {fixed}):
    {variable} = variable
    prob_log = 0
    stimuli_list = np.unique(stimuli).tolist()
    n_actions = np.unique(actions).shape[0]
    Q = dict([[stimulus, np.zeros(n_actions)] for stimulus in stimuli_list])
    for action, reward, stimulus in zip(actions, rewards, stimuli):
        Q[stimulus][action] += alpha * (reward - Q[stimulus][action])
        prob_log += np.log(softmax(Q[stimulus], beta)[action])
    return -prob_log
"""

class RWModel(sciunit.Model, ProducesLoglikelihood):
    """A decision making models (Rescorla–Wagner model)"""

    def __init__(self, paras=None, name=None):
        # if paras != None:
        self.paras = paras
        super().__init__(name=name, paras=paras) # py 3 

    def __make_func(self, *fixed):
        """Create a local negative log-likelihood function for optimization"""
        variable = sorted(set(('alpha', 'beta')).difference(fixed))
        ns = dict() # define namespace
        funcstr = template.format(variable=', '.join(variable), fixed=', '.join(fixed))
        exec(funcstr, ns) # define func in namespace 
        return ns['func']

    def produce_loglikelihood(self, stimuli, rewards):
        #TODO: add error handle
        assert isinstance(self.paras, list)
        f = self.__make_func() # negative log-likelihood
        def logpdf(actions):
            res = 0
            for a, r, s, p in zip(actions, rewards, stimuli, self.paras):
                p_ = list(p.values()) # unpack paras dict
                res -= f(p_, s, r, a)
            return res
        return logpdf
