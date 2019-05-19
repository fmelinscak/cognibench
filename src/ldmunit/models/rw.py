import sciunit
import numpy as np
from ..capabilities import ProducesLoglikelihood
from scipy.optimize import minimize

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
    bounds={'alpha': (1.0e-3, 1.0e3), 
            'beta':  (1.0e-3, 1.0e3)}

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

    def train_with_observations(self, initial_guess, stimuli, rewards, actions, fixed,
                                update_paras=False, verbose=False,):
                                #TODO: add user-defined bounds
                                #TODO: add support for cases not fixed

        # minimize wrt negative log-likelihood function
        assert isinstance(fixed, list)
        res = []
        success = True

        for s, r, a, f in zip(stimuli, rewards, actions, fixed):
            #TODO: assert (s, r, a) has the same n_rows; 
            fixed_var = tuple(f.keys())
            variable = sorted(set(('alpha', 'beta')).difference(fixed_var))
            bounds_var = tuple([self.bounds[i] for i in variable]) # get bounds for variables
            fixed_vals = tuple(f.values())
            vals = (s, r, a, *fixed_vals) if f != None else (s, r, a)

            sol = minimize(self.__make_func(*fixed_var), 
                                    x0=initial_guess, 
                                    args=vals, 
                                    bounds=bounds_var, 
                                    method='L-BFGS-B')

            success = success & sol.success # update marker

            if sol.success:
                optimal_variable = dict(zip(variable, sol.x))
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

                optimal_variable.update(f) # add fixed parametersts
                res.append(optimal_variable)

        if update_paras & success:
            print("Parameters updated")
            self.paras = res
        return res #TODO: add bounds in the returns