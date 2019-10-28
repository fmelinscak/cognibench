import numpy as np

from ldmunit.testing import InteractiveTest, BatchTest
from ldmunit.models import LDMModel
from ldmunit.utils import partialclass
from ldmunit.scores import SmallerBetterScore
from ldmunit.capabilities import PredictsLogpdf


def neg_loglikelihood(actions, predictions):
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
        individual log probabilities for every action-prediction pairs.
    """
    neg_loglike = float(0)
    n_subjects = len(actions)
    for subject_idx in range(n_subjects):
        for act, logprob in zip(actions[subject_idx], predictions[subject_idx]):
            neg_loglike -= logprob(act)
    return neg_loglike


class MSETest(BatchTest):
    """
    Perform batch test on models that produce a real-valued numpy array as its
    predictions. Each prediction is compared to its ground truth using squared error,
    and the final score is computed as the mean squared error.
    """

    # TODO: find a better way to define the feasible range of the score
    score_type = partialclass(SmallerBetterScore, min_score=0, max_score=1)

    def compute_score(self, observation, prediction):
        """
        Compute the mean squared error score from observations and predictions

        Parameters
        ----------
        observation : dict
            Dictionary storing the actions in 'actions' key.

        prediction : numpy.ndarray
            2D numpy array of subject-specific predictions. Each row of the matrix
            stores the predictions for the corresponding subject.

        Returns
        -------
        :class:`ldmunit.scores.SmallerBetterScore`
            Mean squared error.
        """
        action = observation["actions"]
        mse = np.mean((action - prediction) ** 2)
        return self.score_type(mse)


class MAETest(BatchTest):
    """
    Perform batch test on models that produce a real-valued numpy array as its
    predictions. Each prediction is compared to its ground truth using absolute error,
    and the final score is computed as the mean absolute error.
    """

    # TODO: find a better way to define the feasible range of the score
    score_type = partialclass(SmallerBetterScore, min_score=0, max_score=1)

    def compute_score(self, observation, prediction):
        """
        Compute the mean absolute error score from observations and predictions

        Parameters
        ----------
        observation : dict
            Dictionary storing the actions in 'actions' key.

        prediction : numpy.ndarray
            2D numpy array of subject-specific predictions. Each row of the matrix
            stores the predictions for the corresponding subject.

        Returns
        -------
        :class:`ldmunit.scores.SmallerBetterScore`
            Mean absolute error.
        """
        action = observation["actions"]
        mae = np.mean(np.abs(action - prediction))
        return self.score_type(mae)


class CrossEntropyTest(BatchTest):
    """
    Perform batch test on models that produce a real-valued probability numpy array as its
    predictions, meaning each element of the array must be in the range [0, 1].
    The score for one sample is computed as the cross entropy loss H(ground, pred) where ground is the
    ground truth probability vector and pred is the prediction vector output by the model. The final
    score is computed as the mean of these individual cross entropy losses.
    """

    # TODO: find a better way to define the feasible range of the score
    score_type = partialclass(SmallerBetterScore, min_score=0, max_score=100)

    def __init__(self, *args, eps=1e-9, **kwargs):
        self.eps = eps
        super().__init__(*args, **kwargs)

    def compute_score(self, observation, prediction):
        """
        Compute the mean cross entropy between ground truth and model predictions.

        Parameters
        ----------
        observation : dict
            Dictionary storing the actions in 'actions' key.

        prediction : numpy.ndarray
            2D numpy array of subject-specific predictions. Each row of the matrix
            stores the predictions for the corresponding subject.

        Returns
        -------
        :class:`ldmunit.scores.SmallerBetterScore`
            Mean cross entropy error.
        """
        action = observation["actions"]
        prediction_clipped = np.clip(prediction, self.eps, 1 - self.eps)
        N = prediction_clipped.shape[0]
        mean_cross_entropy = -np.sum(action * np.log(prediction_clipped)) / N
        return self.score_type(mean_cross_entropy)


class NLLTest(InteractiveTest):
    """
    Perform interactive test on models that produce a log pdf/pmf as their
    predictions. Negative log-likelihood (NLL) function is used as the score.
    """

    score_type = partialclass(SmallerBetterScore, min_score=0, max_score=1000)
    required_capabilities = InteractiveTest.required_capabilities + (PredictsLogpdf,)

    def compute_score(self, observation, prediction):
        """
        Compute the negative log-likelihood score from observations and predictions

        Parameters
        ----------
        observation : dict
            Dictionary storing the actions in 'actions' key.

        prediction : list of list
            List of subject-specific predictions. Each element is a list and
            stores the predictions for the corresponding subject.

        Returns
        -------
        :class:`ldmunit.scores.SmallerBetterScore`
            Negative log-likelihood.
        """
        nll = neg_loglikelihood(observation["actions"], prediction)
        return self.score_type(nll)


class AICTest(InteractiveTest):
    """
    Perform interactive test on models that produce a log pdf/pmf as their
    predictions. Akaike Information Criterion (AIC) function is used as the score.
    """

    score_type = partialclass(SmallerBetterScore, min_score=0, max_score=1000)
    required_capabilities = InteractiveTest.required_capabilities + (PredictsLogpdf,)

    def generate_prediction(self, multimodel):
        """
        This method simply calls the parent method for the actual functionality
        after storing some necessary variables to compute the score later.

        See Also
        --------
        :func:`InteractiveTest.generate_prediction`
        """
        # save variables necessary to compute score
        self.n_model_params = np.array(
            [len(m.paras) for m in multimodel.subject_models]
        )

        return super().generate_prediction(multimodel)

    def compute_score(self, observation, prediction):
        """
        Compute the Akaike Information Criterion score from observations, predictions
        and, model and input specific parameters stored during generate_prediction.

        Parameters
        ----------
        observation : dict
            Dictionary storing the actions in 'actions' key.

        prediction : list of list
            List of subject-specific predictions. Each element is a list and
            stores the predictions for the corresponding subject.

        Returns
        -------
        :class:`ldmunit.scores.SmallerBetterScore`
            AIC
        """
        nll = neg_loglikelihood(observation["actions"], prediction)
        regularizer = 2 * np.sum(self.n_model_params)
        return self.score_type(nll + regularizer)


class BICTest(InteractiveTest):
    """
    Perform interactive test on models that produce a log pdf/pmf as their
    predictions. Bayesian Information Criterion (BIC) function is used as the score.
    """

    score_type = partialclass(SmallerBetterScore, min_score=0, max_score=1000)
    required_capabilities = InteractiveTest.required_capabilities + (PredictsLogpdf,)

    def generate_prediction(self, multimodel):
        """
        This method simply calls the parent method for the actual functionality
        after storing some necessary variables to compute the score later.

        See Also
        --------
        :func:`InteractiveTest.generate_prediction`
        """
        # save variables necessary to compute score
        stimuli = self.observation["stimuli"]
        self.n_model_params = np.array(
            [len(m.paras) for m in multimodel.subject_models]
        )
        self.n_samples = np.array([len(s) for s in stimuli])

        return super().generate_prediction(multimodel)

    def compute_score(self, observation, prediction):
        """
        Compute the Bayesian Information Criterion score from observations, predictions
        and, model and input specific parameters stored during generate_prediction.

        Parameters
        ----------
        observation : dict
            Dictionary storing the actions in 'actions' key.

        prediction : list of list
            List of subject-specific predictions. Each element is a list and
            stores the predictions for the corresponding subject.

        Returns
        -------
        :class:`ldmunit.scores.SmallerBetterScore`
            BIC
        """
        nll = neg_loglikelihood(observation["actions"], prediction)
        regularizer = np.dot(self.n_model_params, self.n_samples)
        return self.score_type(nll + regularizer)
