import numpy as np
from scipy.stats.mstats import pearsonr
from sciunit import scores
from sciunit import errors
from ldmunit.capabilities import PredictsLogpdf


class BoundedScore(scores.FloatScore):
    def __init__(self, score, *args, min_score, max_score, **kwargs):
        """
        Initialize the score. This class requires two extra mandatory
        keyword-only arguments.

        Parameters
        ----------
        score : float
            Score value.

        min_score : float
            Minimum possible score. This value is used to clip the
            score value when computing norm_score. This is necessary
            to avoid using very small/large values during coloring
            which crashes sciunit. However, this value does not
            affect the original score value in any way.

        max_score : float
            Maximum possible score. This value is used to clip the
            score value when computing norm_score. This is necessary
            to avoid using very small/large values during coloring
            which crashes sciunit. However, this value does not
            affect the original score value in any way.
        """
        super().__init__(score, **kwargs)
        self.min_score = min_score
        self.max_score = max_score

    @property
    def norm_score(self):
        clipped = min(self.max_score, max(self.min_score, self.score))
        return (clipped - self.min_score) / (self.max_score - self.min_score)

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
        if value is not None:
            self.score = value
        return super().color(self.norm_score)


class HigherBetterScore(BoundedScore):
    _description = "Score values where higher is better"


class LowerBetterScore(BoundedScore):
    """
    LowerBetterScore is a score type where lower values are better than
    larger values, similar to an error function. This property is used by
    sciunit library when sorting or color coding the scores.
    """

    _description = "Score values where lower is better"

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
        return 1 - super().norm_score


def _neg_loglikelihood(actions, predictions):
    """
    Compute negative log-likelihood of a multimodel using a collection of
    subject-specific true action and model prediction lists. Each prediction
    list must contain a series of logpdf or logpmf functions.

    Parameters
    ----------
    actions : list of list
        List of subject-specific actions. Each element must be a list
        containing a series of actions.
    predictions : list of list
        List of subject-specific predictions. Each element must be a list
        containing a series of predictions as logpdf or logpmf.

    Returns
    -------
    float
        Negative log-likelihood of the whole multi-subject model on the
        given action and prediction data. It is calculated as the sum of
        individual logpdf/logpmf values for every action-prediction pairs.
    """
    neg_loglike = float(0)
    n_subjects = len(actions)
    for subject_idx in range(n_subjects):
        for act, logpdf in zip(actions[subject_idx], predictions[subject_idx]):
            neg_loglike -= logpdf(act)
    return neg_loglike


class NLLScore(LowerBetterScore):
    required_capabilities = (PredictsLogpdf,)

    @classmethod
    def compute(cls, actions, predictions):
        nll = _neg_loglikelihood(actions, predictions)
        return cls(nll)


class AICScore(LowerBetterScore):
    required_capabilities = (PredictsLogpdf,)

    @classmethod
    def compute(cls, actions, predictions, *args, n_model_params):
        nll = _neg_loglikelihood(actions, predictions)
        regularizer = 2 * np.sum(n_model_params)
        return cls(nll + regularizer)


class BICScore(LowerBetterScore):
    required_capabilities = (PredictsLogpdf,)

    @classmethod
    def compute(cls, actions, predictions, *args, n_model_params, n_samples):
        nll = _neg_loglikelihood(actions, predictions)
        regularizer = np.dot(n_model_params, n_samples)
        return cls(nll + regularizer)


class MSEScore(LowerBetterScore):
    @classmethod
    def compute(cls, actions, predictions):
        mse = np.mean((actions - predictions) ** 2)
        return cls(mse)


class MAEScore(LowerBetterScore):
    @classmethod
    def compute(cls, actions, predictions):
        mae = np.mean(np.abs(actions - predictions))
        return cls(mae)


class PearsonCorrelationScore(HigherBetterScore):
    @classmethod
    def compute(cls, actions, predictions):
        actions = np.asarray(actions).flatten()
        predictions = np.asarray(predictions).flatten()
        corr = pearsonr(np.asarray(actions), np.asarray(predictions))[0]
        return cls(corr)


class CrossEntropyScore(LowerBetterScore):
    @classmethod
    def compute(cls, actions, predictions, *args, eps=1e-9):
        actions = np.asarray(actions)
        predictions_clipped = np.clip(predictions, eps, 1 - eps)
        N = predictions_clipped.shape[0]
        mean_crossent = -np.sum(actions * np.log(predictions_clipped)) / N
        return cls(mean_crossent)
