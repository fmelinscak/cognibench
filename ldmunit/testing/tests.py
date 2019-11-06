import numpy as np

from ldmunit.testing import InteractiveTest, BatchTest
from ldmunit.models import LDMModel
from ldmunit.utils import partialclass
from ldmunit.capabilities import PredictsLogpdf
import ldmunit.scores as scores


class BatchMSETest(BatchTest):
    """
    Perform batch test on models that produce a real-valued numpy array as its
    predictions. Each prediction is compared to its ground truth using squared error,
    and the final score is computed as the mean squared error.
    """

    score_type = partialclass(scores.MSEScore, min_score=0, max_score=1)


class BatchMAETest(BatchTest):
    """
    Perform batch test on models that produce a real-valued numpy array as its
    predictions. Each prediction is compared to its ground truth using absolute error,
    and the final score is computed as the mean absolute error.
    """

    score_type = partialclass(scores.MAEScore, min_score=0, max_score=1)


class BatchCrossEntropyTest(BatchTest):
    """
    Perform batch test on models that produce a real-valued probability numpy array as its
    predictions, meaning each element of the array must be in the range [0, 1].
    The score for one sample is computed as the cross entropy loss H(ground, pred) where ground is the
    ground truth probability vector and pred is the prediction vector output by the model. The final
    score is computed as the mean of these individual cross entropy losses.
    """

    score_type = partialclass(scores.CrossEntropyScore, min_score=0, max_score=1000)

    def __init__(self, *args, eps, **kwargs):
        self.eps = eps
        super().__init__(*args, **kwargs)

    def compute_score(self, *args, **kwargs):
        return super().compute_score(*args, eps=self.eps, **kwargs)


class InteractiveNLLTest(InteractiveTest):
    """
    Perform interactive test on models that produce a log pdf/pmf as their
    predictions. Negative log-likelihood (NLL) function is used as the score.
    """

    score_type = partialclass(scores.NLLScore, min_score=0, max_score=1000)
    required_capabilities = InteractiveTest.required_capabilities + (PredictsLogpdf,)


class InteractiveAICTest(InteractiveTest):
    """
    Perform interactive test on models that produce a log pdf/pmf as their
    predictions. Akaike Information Criterion (AIC) function is used as the score.
    """

    score_type = partialclass(scores.AICScore, min_score=0, max_score=1000)
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

    def compute_score(self, *args, **kwargs):
        return super().compute_score(
            *args, n_model_params=self.n_model_params, **kwargs
        )


class InteractiveBICTest(InteractiveTest):
    """
    Perform interactive test on models that produce a log pdf/pmf as their
    predictions. Bayesian Information Criterion (BIC) function is used as the score.
    """

    score_type = partialclass(scores.BICScore, min_score=0, max_score=1000)
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

    def compute_score(self, *args, **kwargs):
        return super().compute_score(
            *args,
            n_model_params=self.n_model_params,
            n_samples=self.n_samples,
            **kwargs
        )
