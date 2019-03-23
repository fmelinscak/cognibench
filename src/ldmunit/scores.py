import math
import numpy as np

from sciunit import utils
from sciunit import errors
from sciunit import Score
from sciunit.scores import InsufficientDataScore

from .utils import softmax

class MSEScore(Score):
    """A MSE score.

    A float indicating standardized difference
    from a reference mean.
    """

    _allowed_types = (float,)

    _description = ('The difference between the means of the observation and '
                    'prediction divided by the standard deviation of the '
                    'observation')

    @classmethod
    def compute(cls, observation, prediction):
        """Compute a MSE from n observations and predictions."""
        assert isinstance(observation, dict)
        try:
            p_value = prediction['mean']  # Use the prediction's mean
        except (TypeError, KeyError, IndexError):  # If there isn't one...
            try:
                p_value = prediction['value']  # Use the prediction's value.
            except (TypeError, IndexError):  # If there isn't one...
                p_value = prediction  # Use the prediction (assume numeric).

        obs = observation['value']
        try:
            value = np.mean(((obs - p_value) ** 2)) / obs.size
        except (AttributeError):
            obs = np.asarray(obs)
            value = np.mean(((obs - p_value) ** 2)) 
        value = utils.assert_dimensionless(value) #TODO: needed for data? 
        if np.isnan(value):
            score = InsufficientDataScore('One of the input values was NaN')
        else:
            score = MSEScore(value)
        return score

    def __str__(self):
        return 'MSE = %.3f' % self.score

class NLLScore(Score):
    #TODO: Compute log-likelihood by Empirical distribution for unknown case
    """A negative log-likelihood score.
    """

    _allowed_types = (float,)

    @classmethod
    def compute(cls, observation, prediction):
        assert isinstance(observation, dict)
        assert isinstance(prediction, dict)

        # prepare observed value
        try:
            obs = observation['value']
        except (TypeError, KeyError, IndexError):
            try:
                obs = observation['data']
            except (TypeError, IndexError):
                obs = observation

        para = prediction['para'] # return the parameter
        pdf = np.vectorize(prediction['lpdf']) # return a function over np.array positive log-likelihood
        
        value = 0
        value -= pdf(para, obs).sum() # negative
        score = NLLScore(value)
        return score
    
    def __str__(self):
        return 'A negative log likelihood = %.3f' % self.score
