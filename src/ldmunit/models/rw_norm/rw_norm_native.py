import numpy as np
from scipy import stats
from scipy.optimize import minimize
from sciunit.models import Model
from ...capabilities import ProducesLoglikelihood, Trainable

template = """
import numpy as np
from scipy import stats
def func(variable, stimuli, rewards, actions, {fixed}):
    {variable} = variable
    (n_trials, n_features) = stimuli.shape
    w = np.insert(np.zeros_like(stimuli), 0, w0, axis=0)

    mu_pred = np.zeros_like(rewards, dtype=float)
    sd_pred = np.full_like(rewards, sigma, dtype=float)

    for i in range(n_trials):
        # Get current weights and current cues
        w_curr = w[i,:]
        x_curr = stimuli[i,:]

        # Generate outcome prediction
        rhat = np.dot(x_curr, w_curr.T)
        # Predict response
        mu_pred[i] = b0 + b1 * rhat

        # Calculate prediction error based on observed outcome
        pred_err = rewards[i] - rhat

        # Update weights of active cues
        w[i+1,:] = w_curr + alpha * pred_err * x_curr
    pointwise_lpdf = stats.norm.logpdf(actions, mu_pred, sd_pred)
    return -np.sum(pointwise_lpdf)
"""

class RwNormNativeModel(Model, ProducesLoglikelihood, Trainable):
    
    bounds={'alpha': (1.0e-3, 1.0e3), 
            'sigma': (1.0e-3, 1.0e3), 
            'b0':    (1.0e-3, 1.0e3),
            'b1':    (1.0e-3, 1.0e3), 
            'w0':    (1.0e-3, 1.0e3)}

    def __init__(self, paras=None, name=None):
        # if paras != None:
        self.paras = paras
        # super(RwNormNativeModel, self).__init__(name=name, paras=paras)
        super().__init__(name=name, paras=paras) # py 3 

    def __predict_actions(self):
        pass
    
    @staticmethod
    def predict_mu_sd(x, stimuli, rewards):
        alpha, sigma, b0, b1, w0 = x
        (n_trials, n_features) = stimuli.shape
        w = np.insert(np.zeros_like(stimuli), 0, w0, axis=0)

        mu_pred = np.zeros_like(rewards, dtype=float)
        sd_pred = np.full_like(rewards, sigma, dtype=float)

        for i in range(n_trials):
            # Get current weights and current cues
            w_curr = w[i,:]
            x_curr = stimuli[i,:]

            # Generate outcome prediction
            rhat = np.dot(x_curr, w_curr.T)
            # Predict response
            mu_pred[i] = b0 + b1 * rhat

            # Calculate prediction error based on observed outcome
            pred_err = rewards[i] - rhat

            # Update weights of active cues
            w[i+1,:] = w_curr + alpha * pred_err * x_curr
        return mu_pred, sd_pred
    
    def make_func(self, *fixed):
        """Change the fixed parameters for given values"""
        variable = sorted(set(('alpha', 'b0', 'b1', 'sigma', 'w0')).difference(fixed))
        ns = dict() # define namespace
        funcstr = template.format(variable=', '.join(variable), fixed=', '.join(fixed))
        # https://github.com/python/cpython/blob/master/Lib/collections/__init__.py#L421
        exec(funcstr, ns) # define func in namespace 
        return ns['func']

    def train_with_observations(self, initial_guess, stimuli, rewards, actions, fixed,
                                update_paras=False, verbose=False,):
                                #TODO: add bounds for each subj
        res = []
        
        for s, r, a, f in zip(stimuli, rewards, actions, fixed):
            #TODO: assert (s, r, a) has the same n_rows; 
            fixed_var = tuple(f.keys())
            variable = sorted(set(('alpha', 'b0', 'b1', 'sigma', 'w0')).difference(fixed_var))
            bounds_var = tuple([self.bounds[i] for i in variable]) # get bounds for variables
            fixed_vals = tuple(f.values())
            vals = (s, r, a, *fixed_vals)

            sol = minimize(self.make_func(*fixed_var), 
                                    x0=initial_guess, 
                                    args=vals, 
                                    bounds=bounds_var, 
                                    method='L-BFGS-B')
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
                
                optimal_variable.update(f) # add fixed parameters
                res.append(optimal_variable)

        if update_paras:
            print("Parameters updated")
            self.paras = res
        return res #TODO: add bounds in the returns
    
    def produce_loglikelihood(self, stimuli, rewards):
        # assert n_sub
        # assert n_sub = para
        # set paras to n
        # assert len(set(map(len, (a, b, c)))) == 1
        assert len(stimuli) == len(rewards)
        
        mu_pred, sd_pred = [], []
        for s, a, p in zip(stimuli, rewards, self.paras):
            p_ = list(p.values()) # unpack paras dict
            res = self.predict_mu_sd(p_, s, a)
            mu_pred.append(res[0])
            sd_pred.append(res[1])

        def logpdf(actions):
            assert len(stimuli) == len(actions)
            ans = 0
            for a, mu, sd in zip(actions, mu_pred, sd_pred):
                ans += np.sum(stats.norm.logpdf(a, mu, sd))
            return ans

        return logpdf