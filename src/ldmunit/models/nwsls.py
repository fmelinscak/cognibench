import sciunit
import numpy as np
from ..capabilities import ProducesLoglikelihood, Trainable
from scipy.optimize import minimize

class NWSLSModel(sciunit.Model, ProducesLoglikelihood, Trainable):

    bounds={'epsilon': (1.0e-3, 1.0e3)}

    def __init__(self, paras, name=None):
        # if paras != None:
        self.paras = paras
        super().__init__(name=name, paras=paras) # py 3 

    @staticmethod
    def nlll(epsilon, stimuli, rewards, actions):
        """Calculate negative log-likelihood for one subject."""
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

    def train_with_observations(self, initial_guess, stimuli, rewards, actions,
                                update_paras=False, verbose=False,):
                                #TODO: add user-defined bounds
                                #TODO: add support for cases not fixed

        # minimize wrt negative log-likelihood function
        res = []
        success = True

        for s, r, a in zip(stimuli, rewards, actions):
            #TODO: assert (s, r, a) has the same n_rows; 
            variable = ['epsilon']
            bounds_var = tuple([self.bounds[i] for i in variable]) # get bounds for variables
            vals = (s, r, a)

            sol = minimize(self.nlll, 
                                    x0=initial_guess, 
                                    args=vals, 
                                    bounds=bounds_var, 
                                    method='L-BFGS-B')

            success = success & sol.success # update marker

            if sol.success:
                optimal_variable = dict(zip(variable, list(sol.x)))
                if verbose:
                    print("Success, the optimal parameters are:")
                    for k, v in optimal_variable.items():
                        print("{:10} = {:10.4f}".format(k, v))

                optimal_variable.update(f) # add fixed parameters
                res.append(optimal_variable)
            else:
                optimal_variable = dict(zip(variable, initial_guess))
                if verbose:
                    print("Optimal parameters not found, return the initial guess:")
                    for k, v in optimal_variable.items():
                        print("{:8} = {:10.4f}".format(k, v))

                # optimal_variable.update(f) # add fixed parametersts
                res.append(optimal_variable)

        if update_paras & success:
            print("Parameters updated")
            self.paras = res
        return res #TODO: add bounds in the returns