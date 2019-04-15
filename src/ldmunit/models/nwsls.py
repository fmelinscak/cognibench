import sciunit
import numpy as np
from ..capabilities import ProducesLoglikelihood

class NWSLSModel(sciunit.Model, ProducesLoglikelihood):

    def __init__(self, epsilon, name=None):
        self.epsilon = epsilon
        super(NWSLSModel, self).__init__(name=name,
                                          epsilon=epsilon)
 
    def produce_loglikelihood(self, rewards, stimuli):

        def logpdf(actions):

            prob_log = 0
            epsilon = self.epsilon
            prob_list = np.array([1-epsilon/2, epsilon/2, epsilon/2, 1-epsilon/2]).reshape((2,2))

            stimuli_list = np.unique(stimuli).tolist()
            n_actions = np.unique(actions).shape[0]
            
            Q = dict([[stimulus, np.zeros(n_actions)] for stimulus in stimuli_list])
            for action, reward, stimulus in zip(actions, rewards, stimuli):
                idx = 1 if action == reward else 0
                Q[stimulus] = prob_list[idx]
                prob_log += np.log(Q[stimulus][action])
            
            return -prob_log

        return logpdf
