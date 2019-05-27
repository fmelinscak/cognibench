import numpy as np
from sciunit import Score
from sciunit.scores import InsufficientDataScore
from .capabilities import ProducesLoglikelihood

class NLLScore(Score):

    _allowed_types = (float, )

    @classmethod
    def compute(cls, observation, prediction):
        actions = observation['actions']
        # prediction: a callable
        nll = -prediction(actions)
        score = NLLScore(nll)
        return score

    def __str__(self):
        return '{:5.3f}'.format(self.score)

class BICScore(Score):

    _allowed_types = (float, )

    @classmethod
    def compute(cls, observation, prediction):
        actions = observation['actions']
        k = observation['k']
        # prediction: a callable, n_obs
        lpdf, n = prediction
        bic = - lpdf(actions) + k * np.log(n)
        score = BICScore(bic)
        return score

    def __str__(self):
        return '{:5.3f}'.format(self.score)


class AICScore(Score):

    _allowed_types = (float, )

    @classmethod
    def compute(cls, observation, prediction):
        actions = observation['actions']
        k = observation['k']
        # prediction: a callable
        nll = -prediction(actions)
        aic = 2*k + nll
        score = AICScore(aic)
        return score

    def __str__(self):
        return '{:5.3f}'.format(self.score)
