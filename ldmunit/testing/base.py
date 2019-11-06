import numpy as np
from sciunit import Test
from sciunit.errors import Error
from ldmunit.models import LDMModel
from ldmunit.capabilities import Interactive, BatchTrainable


class LDMTest(Test):
    score_type = None

    def __init__(self, *args, score_type=None, **kwargs):
        if score_type is not None:
            self.score_type = score_type
            try:
                score_capabilities = self.score_type.required_capabilities
                self.required_capabilities = (
                    LDMTest.required_capabilities + score_capabilities
                )
            except AttributeError:
                pass
        super().__init__(*args, **kwargs)

    def check_capabilities(self, model, **kwargs):
        if not isinstance(model, LDMModel):
            raise Error(f"Model {model} is not an instance of LDMModel")
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

    required_capabilities = (Interactive,)

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
        stimuli = self.observation["stimuli"]
        rewards = self.observation["rewards"]
        actions = self.observation["actions"]

        predictions = []

        for (
            subject_idx,
            (subject_stimuli, subject_rewards, subject_actions),
        ) in enumerate(zip(stimuli, rewards, actions)):
            multimodel.reset(subject_idx)
            subject_predictions = []
            for s, r, a in zip(subject_stimuli, subject_rewards, subject_actions):
                subject_predictions.append(multimodel.predict(subject_idx, s))
                multimodel.update(subject_idx, s, r, a, False)
            predictions.append(subject_predictions)
        return predictions

    def compute_score(self, observation, predictions, **kwargs):
        actions = observation["actions"]
        return self.score_type.compute(actions, predictions, **kwargs)


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
        stimuli = self.observation["stimuli"]

        predictions = []
        for s in stimuli:
            predictions.append(model.predict(s))
        return predictions

    def compute_score(self, observation, predictions, **kwargs):
        actions = observation["actions"]
        return self.score_type.compute(actions, predictions, **kwargs)


class BatchTrainAndTest(LDMTest):
    required_capabilities = (BatchTrainable,)

    def __init__(
        self,
        *args,
        train_percentage=0.75,
        seed=None,
        train_indices=None,
        test_indices=None,
        **kwargs,
    ):
        """
        If train_indices is given, it is used; else, a random train/test split is used.
        """
        assert (
            train_percentage > 0 and train_percentage < 1
        ), "train_percentage must be in range (0, 1)"
        super().__init__(*args, **kwargs)
        if train_indices is None:
            assert test_indices is None
            n_obs = len(self.observation["stimuli"])
            indices = np.arange(n_obs, dtype=np.int64)
            np.random.RandomState(seed).shuffle(indices)
            n_train = round(n_obs * train_percentage)
            train_indices = indices[:n_train]
            test_indices = indices[n_train:]
        else:
            assert test_indices is not None
            train_indices = train_indices
            test_indices = test_indices

        self.train_observation = dict()
        self.train_observation["stimuli"] = self.observation["stimuli"][train_indices]
        self.train_observation["actions"] = self.observation["actions"][train_indices]
        self.observation["stimuli"] = self.observation["stimuli"][test_indices]
        self.observation["actions"] = self.observation["actions"][test_indices]

    def generate_prediction(self, model):
        model.fit(self.train_observation["stimuli"], self.train_observation["actions"])
        predictions = []
        for s in self.observation["stimuli"]:
            predictions.append(model.predict(s))
        return predictions

    def compute_score(self, observation, predictions, **kwargs):
        actions = observation["actions"]
        return self.score_type.compute(actions, predictions, **kwargs)
