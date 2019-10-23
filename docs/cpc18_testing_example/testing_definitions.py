from ldmunit.utils import partialclass
from ldmunit.scores import SmallerBetterScore
from ldmunit.capabilities import Interactive, ContinuousAction
from ldmunit.tests import InteractiveTest
from ldmunit.models import LDMModel
import numpy as np

import importlib

#####################################################
# 1. We define the interactive test that scores the
# models according to MSE
#####################################################

class MSETest(InteractiveTest):
    """
    Perform interactive test and produce mean squared error as the score.
    """

    score_type = partialclass(SmallerBetterScore, min_score=0, max_score=1)
    required_capabilities = InteractiveTest.required_capabilities + (ContinuousAction, )

    def compute_score(self, observation, prediction):
        """
        Compute MSE score from observations and predictions

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
            Mean squared error (MSE).
        """
        mse = 0.0
        action = observation['actions']
        n_total_actions = 0
        for subject_acts, subject_pred in zip(action, prediction):
            n_total_actions += len(subject_acts)
            for act, pred in zip(subject_acts, subject_pred):
                mse += np.sum((act - pred)**2)
        mse /= n_total_actions

        return self.score_type(mse)


#####################################################
# 2. We also define the model specification submitted
# by multiple contestants. In this example, every
# contestant are required to define a wrapper
# function CPC18_BEASTsd_pred around their model
# implementation.
#
# For simplicity's sake, the different models in this
# example only differ in some parameter values.
#####################################################


class BEASTsdModel(LDMModel, Interactive, ContinuousAction):
    """
    """
    name = "BEASTsd"

    def __init__(self, *args, import_base_path, **kwargs):
        import_file = '{}.CPC18_BEASTsd_pred'.format(import_base_path)
        self.module = importlib.import_module(import_file)
        super().__init__(*args, **kwargs)

    def reset(self):
        pass

    def predict(self, stimulus):
        """
        """
        return self.module.CPC18_BEASTsd_pred(*stimulus)

    def act(self, stimulus):
        """
        """
        pass

    def update(self, stimulus, reward, action, done):
        pass
