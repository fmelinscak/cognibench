import numpy as np
from scipy.stats.mstats import pearsonr
from sciunit import scores
from sciunit import errors
from ldmunit.capabilities import PredictsLogpdf, ReturnsNumParams


class BoundedScore(scores.FloatScore):
    def __init__(self, score, *args, min_score, max_score, **kwargs):
        """
        Initialize the score. This class requires two mandatory
        keyword-only arguments.

        Parameters
        ----------
        score : float
            Score value.

        min_score : float
            This value is used to clip the score value when coloring
            the scores in a notebook environment. This is necessary to
            avoid using very small/large values during coloring which
            crashes sciunit. However, this value does not affect the
            original score value or their ordering in any way.

        max_score : float
            This value is used to clip the score value when coloring
            the scores in a notebook environment. This is necessary to
            avoid using very small/large values during coloring which
            crashes sciunit. However, this value does not affect the
            original score value or their ordering in any way.
        """
        super().__init__(score, **kwargs)
        self.min_score = min_score
        self.max_score = max_score


class HigherBetterScore(BoundedScore):
    _description = "Score values where higher is better"

    def color(self, value=None):
        """
        Ensure that a normalized value is passed to parent class' color method which
        does the real work.

        Parameters
        ----------
        value : float
            Score value to color. If None, function uses `self.score`

        See Also
        --------
        :py:mod:`sciunit.scores`
        """
        clipped = min(self.max_score, max(self.min_score, self.score))
        normalized = (clipped - self.min_score) / (self.max_score - self.min_score)
        return super().color(normalized)


class LowerBetterScore(BoundedScore):
    """
    LowerBetterScore is a score type where lower values are better than
    larger values (e.g. mean squared error). This property is used by
    sciunit library when sorting or color coding the scores.
    """

    _description = "Score values where lower is better"

    def color(self, value=None):
        """
        Ensure that a normalized value is passed to parent class' color method which
        does the real work.

        Parameters
        ----------
        value : float
            Score value to color. If None, function uses `self.score`

        See Also
        --------
        :py:mod:`sciunit.scores`
        """
        neg_min = -self.min_score
        neg_max = -self.max_score
        clipped = max(neg_max, min(neg_min, -self.score))
        normalized = (clipped - neg_min) / (neg_max - neg_min)
        return super().color(normalized)

    @property
    def norm_score(self):
        """
        Used for sorting. Lower is better.

        Returns
        -------
        float
            Score value normalized to 0-1 range computed by clipping self.score to the
            min/max range and then transforming to a value in [0, 1].
        """
        return -super().norm_score


def _neg_loglikelihood(actions, predictions):
    """
    Compute negative log-likelihood of a series of actions and logpdf/logpmf predictions.

    Parameters
    ----------
    actions : array-like
        Sequence of actions.
    predictions : array-like
        Sequence of logpdf/logpmf predictions. For an action `a` and prediction `P`, logpdf/logpmf
        value at a must be equal to `P(a)`.

    Returns
    -------
    float
        Negative log-likelihood.
    """
    neg_loglike = float(0)
    for act, logpdf in zip(actions, predictions):
        neg_loglike -= logpdf(act)
    return neg_loglike


class NLLScore(LowerBetterScore):
    """
    Negative log-likelihood score object.

    This score object requires a corresponding test model to predict logpdf (or logpmf).
    """

    required_capabilities = (PredictsLogpdf,)

    @classmethod
    def compute(cls, actions, predictions):
        """
        Return NLL score as a Score object from a sequence of actions
        and logpdf/logpmf predictions.
        """
        nll = _neg_loglikelihood(actions, predictions)
        return cls(nll)


class AICScore(LowerBetterScore):
    """
    Akaike Information Criterion score object.

    This score object requires a corresponding test model
      - to predict logpdf (or logpmf),
      - to be able to return its number of parameters.
    """

    required_capabilities = (PredictsLogpdf, ReturnsNumParams)

    @classmethod
    def compute(cls, actions, predictions, *args, n_model_params):
        """
        Return AIC score as a Score object from a sequence of actions
        and logpdf/logpmf predictions.
        """
        nll = _neg_loglikelihood(actions, predictions)
        regularizer = 2 * np.sum(n_model_params)
        return cls(nll + regularizer)


class BICScore(LowerBetterScore):
    """
    Bayesian Information Criterion score object.

    This score object requires a corresponding test model
      - to predict logpdf (or logpmf),
      - to be able to return its number of parameters.
    """

    required_capabilities = (PredictsLogpdf, ReturnsNumParams)

    @classmethod
    def compute(cls, actions, predictions, *args, n_model_params, n_samples):
        """
        Return BIC score as a Score object from a sequence of actions
        and logpdf/logpmf predictions.
        """
        nll = _neg_loglikelihood(actions, predictions)
        regularizer = np.dot(n_model_params, n_samples)
        return cls(nll + regularizer)


class MSEScore(LowerBetterScore):
    """
    Mean squared error score object.
    """

    @classmethod
    def compute(cls, actions, predictions):
        """
        Compute the score from a sequence of actions and predictions. Each
        action and prediction may have arbitrary dimensions; in any case, the mean
        is taken over all dimensions.
        """
        mse = np.mean((actions - predictions) ** 2)
        return cls(mse)


class MAEScore(LowerBetterScore):
    """
    Mean absolute error score object.
    """

    @classmethod
    def compute(cls, actions, predictions):
        """
        Compute the score from a sequence of actions and predictions. Each
        action and prediction may have arbitrary dimensions; in any case, the mean
        is taken over all dimensions.
        """
        mae = np.mean(np.abs(actions - predictions))
        return cls(mae)


class PearsonCorrelationScore(HigherBetterScore):
    """
    Pearson correlation coefficient score object.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, min_score=-1, max_score=1, **kwargs)

    @classmethod
    def compute(cls, actions, predictions):
        """
        Each action and prediction is assumed to be a scalar value.
        """
        actions = np.asarray(actions).flatten()
        predictions = np.asarray(predictions).flatten()
        corr = pearsonr(np.asarray(actions), np.asarray(predictions))[0]
        return cls(corr)


class CrossEntropyScore(LowerBetterScore):
    """
    Cross-entropy score object.
    """

    @classmethod
    def compute(cls, actions, predictions, *args, eps=1e-9):
        actions = np.asarray(actions)
        predictions_clipped = np.clip(predictions, eps, 1 - eps)
        N = predictions_clipped.shape[0]
        mean_crossent = -np.sum(actions * np.log(predictions_clipped)) / N
        return cls(mean_crossent)


class AccuracyScore(HigherBetterScore):
    """
    Accuracy score object.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, min_score=0.0, max_score=1.0, **kwargs)

    @classmethod
    def compute(cls, actions, predictions):
        """
        Returned accuracy is between 0.0 and 1.0
        """
        actions = np.asarray(actions)
        predictions = np.asarray(predictions)
        assert len(actions.shape) == 1 and actions.shape == predictions.shape
        n_correct = np.sum(actions == predictions)
        return cls(float(n_correct) / len(actions))
