import numpy as np
from scipy import stats
from sciunit.models import Model
from ...capabilities import ProducesLoglikelihood
import inspect
import os
import rpy2
from rpy2.robjects import r, numpy2ri, packages

class RwNormRModel(Model, ProducesLoglikelihood):

    def __init__(self, alpha, sigma, b0, b1, w0=None, name=None):
        if w0 is None:
            w0 = 0.0
        self.alpha = alpha
        self.sigma = sigma
        self.b0 = b0
        self.b1 = b1
        self.w0 = w0
        super(RwNormRModel, self).__init__(name=name,
                                          alpha=alpha, sigma=sigma,
                                          b0=b0, b1=b1, w0=w0)

    def produce_loglikelihood(self, stimuli, rewards):
        (n_trials, n_features) = stimuli.shape
        class_path = os.path.dirname(inspect.getfile(type(self)))

        r_path = os.path.join(class_path, 'rw_norm_predict.R')
        with open(r_path, 'r') as f:
            string = f.read()
        predict = packages.STAP(string, "rw_norm_predict")

        # conversion from numpy
        numpy2ri.activate()
        stimuli_ = r.matrix(stimuli, nrow=n_trials, ncol=n_features)
        rewards_ = r.matrix(rewards, nrow=n_trials)
        res = predict.rw_norm_predict(stimuli_, rewards_, 
            self.alpha, self.sigma, self.b0, self.b1, self.w0)
        numpy2ri.deactivate()

        mu_pred = np.asarray(res[0]).squeeze()
        sd_pred = np.asarray(res[1]).squeeze()
        
        # Create logpdf
        def logpdf(actions):
            pointwise_logpdf = stats.norm.logpdf(actions,
                                                 loc=mu_pred, scale=sd_pred)
            return np.sum(pointwise_logpdf)

        return logpdf
