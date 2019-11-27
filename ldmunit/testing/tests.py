import numpy as np

from ldmunit.testing import InteractiveTest, BatchTest
from ldmunit.models import LDMModel
from ldmunit.utils import partialclass
from ldmunit.capabilities import PredictsLogpdf, ReturnsNumParams
import ldmunit.scores as scores


class InteractiveAICTest(InteractiveTest):
    """
    Perform interactive test on models that produce a log pdf/pmf as their
    predictions. Akaike Information Criterion (AIC) function is used as the score.
    """

    score_type = partialclass(scores.AICScore, min_score=0, max_score=1000)
    required_capabilities = InteractiveTest.required_capabilities + (
        PredictsLogpdf,
        ReturnsNumParams,
    )

    def predict_single(self, model, observations, **kwargs):
        """
        This method simply calls the parent method for the actual functionality
        after storing some necessary variables to compute the score later.

        See Also
        --------
        :func:`InteractiveTest.generate_prediction`
        """
        # save variables necessary to compute score
        self.n_model_params = model.n_params()

        return super().predict_single(model, observations)

    def compute_score_single(self, *args, **kwargs):
        return super().compute_score_single(
            *args, n_model_params=self.n_model_params, **kwargs
        )


class InteractiveBICTest(InteractiveTest):
    """
    Perform interactive test on models that produce a log pdf/pmf as their
    predictions. Bayesian Information Criterion (BIC) function is used as the score.
    """

    score_type = partialclass(scores.BICScore, min_score=0, max_score=1000)
    required_capabilities = InteractiveTest.required_capabilities + (
        PredictsLogpdf,
        ReturnsNumParams,
    )

    def predict_single(self, model, observations, **kwargs):
        """
        This method simply calls the parent method for the actual functionality
        after storing some necessary variables to compute the score later.

        See Also
        --------
        :func:`InteractiveTest.generate_prediction`
        """
        # save variables necessary to compute score
        stimuli = observations["stimuli"]
        self.n_model_params = model.n_params()
        self.n_samples = len(stimuli)

        return super().predict_single(model, observations)

    def compute_score_single(self, *args, **kwargs):
        return super().compute_score_single(
            *args,
            n_model_params=self.n_model_params,
            n_samples=self.n_samples,
            **kwargs
        )
