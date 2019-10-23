from sciunit import Test
from sciunit.errors import Error
from ldmunit.models import LDMModel
from ldmunit.capabilities import Interactive


class LDMTest(Test):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def check_capabilities(self, model, **kwargs):
        if not isinstance(model, LDMModel):
            raise Error(f'Model {model} is not an instance of LDMModel')
        super().check_capabilities(model, **kwargs)


class InteractiveTest(LDMTest):
    """
    Perform interactive tests by feeding the input samples (stimuli) one at a
    time. This class is not intended to be used directly since it does not
    specify how the score should be computed. In order to create concrete
    interactive tests, create a subclass and specify how the score should be
    computed.

    See Also
    --------
    :class:`NLLTest`, :class:`AICTest`, :class:`BICTest` for examples of concrete interactive test classes
    """
    required_capabilities = (Interactive, )

    def __init__(self, *args, **kwargs):
        """
        Other Parameters
        ----------------
        **kwargs : any type
            All the keyword arguments are passed to `__init__` method of :class:`sciunit.tests.Test`.
            `observation` dictionary must contain 'stimuli', 'rewards' and 'actions' keys.
            Value for each these keys must be a list of list (or any other iterable) where
            outer list is over subjects and inner list is over samples.

        See Also
        --------
        :py:meth:`InteractiveTest.generate_prediction`
        """
        super().__init__(*args, **kwargs)

    def generate_prediction(self, multimodel):
        """
        Generate predictions from a multi-subject model one at a time.

        Parameters
        ----------
        multimodel : :class:`ldmunit.models.LDMModel` and :class:`ldmunit.capabilities.Interactive`
            Multi-subject model

        Returns
        -------
        list of list
            Predictions
        """
        stimuli = self.observation['stimuli']
        rewards = self.observation['rewards']
        actions = self.observation['actions']

        predictions = []

        for subject_idx, (subject_stimuli, subject_rewards, subject_actions) in enumerate(zip(stimuli, rewards, actions)):
            multimodel.reset(subject_idx)
            subject_predictions = []
            for s, r, a in zip(subject_stimuli, subject_rewards, subject_actions):
                subject_predictions.append(multimodel.predict(subject_idx, s))
                multimodel.update(subject_idx, s, r, a, False)
            predictions.append(subject_predictions)
        return predictions


class BatchTest(LDMTest):
    def __init__(self, *args, **kwargs):
        """
        Perform batch tests by predicting the outcome for each input sample without
        doing any model update. This class is not intended to be used directly since
        it does not specify how the score should be computed. In order to create
        concrete batch tests, create a subclass and specify how the score
        should be computed.

        Other Parameters
        ----------------
        **kwargs : any type
            All the keyword arguments are passed to `__init__` method of :class:`sciunit.tests.Test`.
            `observation` dictionary must contain 'stimuli', and 'actions' keys.
            Value for each these keys must be a list of 'stimuli' resp. 'action'.

        See Also
        --------
        :py:meth:`BatchTest.generate_prediction`
        """
        super().__init__(*args, **kwargs)

    def generate_prediction(self, model):
        """
        Generate predictions from a given model

        Parameters
        ----------
        model : :class:`ldmunit.models.LDMModel`
            Model to test

        Returns
        -------
        list
            Predictions
        """
        stimuli = self.observation['stimuli']

        predictions = []
        for s in stimuli:
            predictions.append(model.predict(s))
        return predictions
