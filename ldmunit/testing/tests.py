import numpy as np
from .base import LDMTest
from ldmunit.capabilities import Interactive
from overrides import overrides


class InteractiveTest(LDMTest):
    """
    Perform interactive tests by feeding the input samples (stimuli) one at a
    time and updating the model after each sample with the corresponding reward.
    """

    required_capabilities = (Interactive,)

    @overrides
    def __init__(self, *args, **kwargs):
        """
        Parameters
        ----------
        observation
            In single-subject case, the dictionary must contain 'stimuli', 'rewards' and 'actions' keys.

        See Also
        --------
        :class:`ldmunit.testing.LDMTest` for a description of the arguments.
        """
        super().__init__(*args, **kwargs)

    @overrides
    def predict_single(self, model, observations, **kwargs):
        stimuli = observations["stimuli"]
        rewards = observations["rewards"]
        actions = observations["actions"]

        # TODO: this logic is also in ReinforcementLearningFittingMixin.
        predictions = []
        model.reset()
        for s, r, a in zip(stimuli, rewards, actions):
            predictions.append(model.predict(s))
            model.update(s, r, a, False)
        return predictions

    @overrides
    def compute_score_single(self, observations, predictions, **kwargs):
        return self.score_type.compute(observations["actions"], predictions, **kwargs)


class BatchTest(LDMTest):
    """
    BatchTest class allows passing the stimuli-action pairs to the model in a single batch instead of
    performing interactive testing. `predict` method of a model used in this testing method must accept
    a sequence of stimuli, not just one stimulus.
    """

    @overrides
    def __init__(self, *args, **kwargs):
        """
        Parameters
        ----------
        observation
            In single-subject case, the dictionary must contain 'stimuli' and 'actions' keys.

        See Also
        --------
        :class:`ldmunit.testing.LDMTest` for a description of the arguments.
        """
        super().__init__(*args, **kwargs)

    @overrides
    def predict_single(self, model, observations, **kwargs):
        return model.predict(observations["stimuli"])

    @overrides
    def compute_score_single(self, observations, predictions, **kwargs):
        return self.score_type.compute(observations["actions"], predictions, **kwargs)


class BatchTestWithSplit(BatchTest):
    """
    Testing class that allows specifying training and testing stimulus-action pairs to be specified separately
    for each subject. This is in contrast to the standard :class:`LDMTest` class where models are optimized and
    tested on the same stimulus-action pairs.
    """

    @overrides
    def __init__(self, *args, **kwargs):
        """
        Parameters
        ----------
        observation
            In single-subject case, the dictionary must contain 'train' and 'test' keys. Each of `observation['train']`
            and `observation['test']` should contain 'stimuli' and 'action' keys.

            The model will first be fitted to the stimuli-action pairs given in 'train', and then the
            predictions for the pairs in 'test' will be generated.

        See Also
        --------
        :class:`ldmunit.testing.LDMTest` for a description of the arguments.
        """
        kwargs["optimize_models"] = True
        super().__init__(*args, **kwargs)

    @overrides
    def get_fitting_observations_single(self, dictionary):
        return dictionary["train"]

    @overrides
    def get_testing_observations_single(self, dictionary):
        return dictionary["test"]
