import numpy as np
from scipy import stats
from sciunit.models import Model
from ...capabilities import ProducesLoglikelihood

import pandas as pd


class RwNormNativeModel(Model, ProducesLoglikelihood):

    def __init__(self, paras=None, name=None):
        #TODO: init w/o paras
        assert isinstance(paras, dict)
        self.alpha = paras['alpha']
        self.sigma = paras['sigma']
        self.b0 = paras['b0']
        self.b1 = paras['b1']
        self.w0 = paras['w0']
        self.paras = paras
        super().__init__(name=name, paras=paras)

        
    def __produce_loglikelihood(self, w0, b0, b1, alpha, sigma, stimuli, rewards):
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

    def produce_loglikelihood(self, stimuli, rewards, paras=None): #TODO: better error handling for pandas
        assert isinstance(rewards, pd.DataFrame)
        assert isinstance(stimuli, pd.DataFrame)

        # prep paras
        if paras == None:
            paras = self.paras
            alpha = paras['alpha']
            sigma = paras['sigma']
            b0 = paras['b0']
            b1 = paras['b1']
            w0 = paras['w0']

        sub_val = stimuli['subject'].value_counts().sort_index()
        mu_pred = list() #TODO: add support for subject index other than int, e.g. 'SUB001' 
        sd_pred = list()
        for i in sub_val.index:
            stimuli_ = stimuli[stimuli['subject'] == i].drop(columns="subject")
            rewards_ = rewards[rewards['subject'] == i].drop(columns="subject")
            res = self.__produce_loglikelihood(w0[i], b0[i], b1[i], alpha[i], sigma[i], stimuli_.values, rewards_.values)
            mu_pred.append(res[0])
            sd_pred.append(res[1])

        # Create logpdf
        def logpdf(actions):
            assert isinstance(actions, pd.DataFrame)

            ans = 0
            sub_val = actions['subject'].value_counts().sort_index()

            for i in sub_val.index:
                actions_ = actions[actions['subject'] == i].drop(columns="subject")
                #ERROR: always return stats.norm.logpdf(action s_.values)
                ans += np.sum(stats.norm.logpdf(actions_.values, mu_pred[i], sd_pred[i]))

            return ans

        return logpdf
