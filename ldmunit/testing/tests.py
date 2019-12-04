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


class BatchTrainAndTest(LDMTest):
    @overrides
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
        Parameters
        ----------
        observation
            In single-subject case, the dictionary must contain 'stimuli' and 'actions' keys.

        train_percentage : float
            Percentage of randomly selected input samples to use for training. Used only if `train_indices` and
            `test_indices` are not given. Must be between 0 and 1.

        seed : int
            Seed to use for randomly selecting training samples.

        train_indices : np.array
            Indices of the training samples to use instead of randomly choosing them.

        test_indices : np.array
            Indices of the testing samples to use instead of randomly choosing them. Must be given if `train_indices`
            is specified.

        See Also
        --------
        :class:`ldmunit.testing.LDMTest` for a description of the arguments.
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
            self.train_indices = indices[:n_train]
            self.test_indices = indices[n_train:]
        else:
            assert test_indices is not None
            self.train_indices = train_indices
            self.test_indices = test_indices

    @overrides
    def predict_single(self, model, observations, **kwargs):
        x_train = observations["stimuli"][self.train_indices]
        y_train = observations["actions"][self.train_indices]
        model.fit(x_train, y_train)

        x_test = observations["stimuli"][self.test_indices]
        predictions = np.asarray(model.predict(x_test))

        return predictions

    @overrides
    def compute_score_single(self, observations, predictions, **kwargs):
        actions = observations["actions"][self.test_indices]
        return self.score_type.compute(actions, predictions, **kwargs)

    @overrides
    def persist_predictions(self, path, predictions):
        """
        Save the train and test indices in addition to the predictions.
        """
        train_indices_path = f"{path}_train_indices"
        test_indices_path = f"{path}_test_indices"
        np.save(train_indices_path, self.train_indices)
        np.save(test_indices_path, self.test_indices)
        if self.logging > 1:
            print(f"Train indices are saved in {train_indices_path}")
            print(f"Test indices are saved in {test_indices_path}")
        super().persist_predictions(path, predictions)
