import math
import numpy as np

from sciunit import utils
from sciunit import errors
from sciunit import Score
from sciunit.scores import InsufficientDataScore

class NLLScore(Score):
    #TODO: Compute log-likelihood by Empirical distribution for unknown case
    """A negative log-likelihood score.
    """

    _allowed_types = (float,)

    @classmethod
    def compute(cls, observation, prediction):
        """
        observation: dict.
            Dict containing stimuli, rewards and actions.
        prediction: callables.
            The log-likehood function.
        """
        actions = observation['actions']
        value = prediction(actions) # prediction: a callable
        score = NLLScore(value)
        return score
    
    def __str__(self):
        return 'A negative log likelihood = %.3f' % self.score