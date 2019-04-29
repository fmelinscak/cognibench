import numpy as np
from scipy import stats
from sciunit.models import Model
from ...capabilities import ProducesLoglikelihood


class RwNormNativeModel(Model, ProducesLoglikelihood):

    def __init__(self, alpha, sigma, b0, b1, w0=None, name=None):
        if w0 is None:
            w0 = 0.0
        self.alpha = alpha
        self.sigma = sigma
        self.b0 = b0
        self.b1 = b1
        self.w0 = w0
        super(RwNormNativeModel, self).__init__(name=name,
                                          alpha=alpha, sigma=sigma,
                                          b0=b0, b1=b1, w0=w0)

    def produce_loglikelihood(self, stimuli, rewards):
        (n_trials, n_features) = stimuli.shape
        w = np.zeros((n_trials + 1, n_features))
        w[0,:] = self.w0

        mu_pred = np.zeros((n_trials,))
        sd_pred = np.full((n_trials,), self.sigma)

        for i in range(n_trials):
            # Get current weights and current cues
            w_curr = w[i,:]
            x_curr = stimuli[i,:]
            
            # Generate outcome prediction
            rhat = np.dot(x_curr, w_curr.T)
            # Predict response
            mu_pred[i] = self.b0 + self.b1 * rhat
            
            # Calculate prediction error based on observed outcome
            pred_err = rewards[i] - rhat
            
            # Update weights of active cues
            w[i+1,:] = w_curr + self.alpha * pred_err * x_curr
        # Create logpdf
        def logpdf(actions):
            pointwise_logpdf = stats.norm.logpdf(actions,
                                                 loc=mu_pred, scale=sd_pred)
            return np.sum(pointwise_logpdf)

        return logpdf
