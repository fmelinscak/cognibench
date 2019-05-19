import sciunit
import numpy as np
from ..capabilities import ProducesLoglikelihood

class NWSLSModel(sciunit.Model, ProducesLoglikelihood):

    def __init__(self, paras, name=None):
        # if paras != None:
        self.paras = paras
        super().__init__(name=name, paras=paras) # py 3 

    @staticmethod
    def nlll(epsilon, stimuli, rewards, actions):
        prob_log = 0
        prob_list = np.array([1-epsilon/2, epsilon/2, epsilon/2, 1-epsilon/2]).reshape((2,2))

        stimuli_list = np.unique(stimuli).tolist()
        n_actions = np.unique(actions).shape[0]
        
        Q = dict([[stimulus, np.zeros(n_actions)] for stimulus in stimuli_list])
        for action, reward, stimulus in zip(actions, rewards, stimuli):
            idx = 1 if action == reward else 0
            Q[stimulus] = prob_list[idx]
            prob_log += np.log(Q[stimulus][action])
        return -prob_log

    def produce_loglikelihood(self, stimuli, rewards):
        #TODO: add error handle
        assert isinstance(self.paras, list)
        def logpdf(actions):
            res = 0
            for a, r, s, p in zip(actions, rewards, stimuli, self.paras):
                p_ = list(p.values()) # unpack paras dict
                res -= self.nlll(p_[0], s, r, a)
            return res
        return logpdf
