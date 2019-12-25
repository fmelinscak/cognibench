from os import makedirs
import traceback
from os.path import join as pathjoin
import numpy as np
from sciunit import Test as SciunitTest, Score as SciunitScore
from sciunit.errors import Error
from ldmunit.models import LDMModel
from ldmunit.capabilities import MultiSubjectModel
from ldmunit.models.utils import single_from_multi_obj, reverse_single_from_multi_obj
from overrides import overrides
from ldmunit.logging import logger
from collections import defaultdict


_MULTI_LIST_KEY = "__list"


class LDMTest(SciunitTest):
    """
    Base test class for all LDMUnit tests.

    This class defines the common functionality that can be used by further testing classes. In addition to sciunit
    interaction, this class defines the multi-subject testing framework, and requires deriving classes to only define
    single-subject testing logic. This class can not be used directly. The deriving classes should implement at least `predict_single` and `compute_score_single` methods
    to define the testing procedure. This class only accepts models that are subclasses of :class:`ldmunit.models.LDMModel`.
    """

    score_type = None

    @overrides
    def __init__(
        self,
        observation,
        *args,
        score_type=None,
        multi_subject=False,
        score_aggr_fn=np.mean,
        persist_path=None,
        fn_kwargs_for_score=None,
        optimize_models=True,
        **kwargs,
    ):
        """
        Parameters
        ----------
        observation : dict or list of dict
            In a single-subject test, this dictionary must contain the data for testing. The exact keys of the dictionary
            is determined by the concrete test classes.

            In a multi-subject test, this is a sequence where each element is a dictionary storing the data for the
            corresponding subject. Similarly, exact keys are left to the concrete test classes.

        score_type : :class:`sciunit.Score`
            A sciunit Score class (not object). See `ldmunit.scores` for several possibilities. The score type can define its own
            `required_capabilities` class field. In that case, these capabilities will be appended to the required
            capabilities of the test class.

        multi_subject : bool
            Whether the data and the models are multi-subject.

            If `True`, the data is expected to be a sequence where each element is the dictionary of the corresponding subject. Similarly, the model
            should be a multi-subject model (see :py:function:`ldmunit.models.utils.multi_from_single_cls` and :class:`ldmunit.capabilities.MultiSubjectModel`).

            If `False`, data is expected to be a dictionary as usual, and the model should be a standard single-subject
            model.

        score_aggr_fn : callable
            If a multi-subject test is performed, this callable defines how to combine the score values of different
            subjects to compute the final score of the test. Signature is as below (e.g. :py:function:`numpy.mean`):

            `score_aggr_fn(Sequence[float]) -> float`

        persist_path : str (Optional)
            Path to the folder where test logs such as predictions, scores and models will be saved. Directory
            is automatically created if it does not exist. If `None`, no logs will be persisted.

        fn_kwargs_for_score : callable
            Callable to generate required keyword arguments for the score computation, if necessary. Some score objects
            require more than just the predictions and the observations to be computed, such as AIC or BIC. In that case
            this function can be used to generate a dictionary to provide these arguments to such score computing functions.
            The signature is as below:

            `fn_kwargs_for_score(single_subj_model, single_subj_observations, single_subj_predictions) -> dict`

        optimize_models : bool
            If `True`, passed models' `fit` method will be called on the given observation data before generating the
            predictions. If `False`, no model fitting is performed.
        """
        self.multi_subject = multi_subject
        self.score_aggr_fn = score_aggr_fn
        self.persist_path = persist_path
        self.fn_kwargs_for_score = fn_kwargs_for_score
        self.optimize_models = optimize_models

        if multi_subject:
            assert isinstance(observation, list)
            # required to make observation variable play well with sciunit
            observation = {_MULTI_LIST_KEY: observation}

        if score_type is not None:
            assert issubclass(score_type, SciunitScore)
            self.score_type = score_type
            try:
                score_capabilities = self.score_type.required_capabilities
                self.required_capabilities = (
                    LDMTest.required_capabilities + score_capabilities
                )
            except AttributeError:
                pass
        super().__init__(observation, *args, **kwargs)

    @overrides
    def check_capabilities(self, model, **kwargs):
        """
        Check if the passed model is derived from :class:`ldmunit.models.LDMModel` and then delegate the rest of the
        checking to sciunit framework.
        """
        if not isinstance(model, LDMModel):
            raise Error(f"Model {model} is not an instance of LDMModel")
        super().check_capabilities(model, **kwargs)

    def get_fitting_observations_single(self, dictionary):
        """
        Return part of the single subject observation dictionary that will be used for model fitting.

        Parameters
        ----------
        dictionary : dict
            Dictionary containing observations for a single subject.

        Returns
        -------
        out : dict
            Dictionary containing fitting observations.
        """
        return dictionary

    def get_testing_observations_single(self, dictionary):
        """
        Return part of the single subject observation dictionary that will be used for model testing.

        Parameters
        ----------
        dictionary : dict
            Dictionary containing observations for a single subject.

        Returns
        -------
        out : dict
            Dictionary containing testing observations.
        """
        return dictionary

    def get_fitting_observations(self):
        """
        Get the fitting part of `self.observation` variable.
        """
        if self.multi_subject:
            out = [
                self.get_fitting_observations_single(x)
                for x in self.observation[_MULTI_LIST_KEY]
            ]
            return out
        else:
            return self.get_fitting_observations_single(self.observation)

    def get_testing_observations(self):
        """
        Get the testing part of `self.observation` variable.
        """
        if self.multi_subject:
            out = [
                self.get_testing_observations_single(x)
                for x in self.observation[_MULTI_LIST_KEY]
            ]
            return out
        else:
            return self.get_testing_observations_single(self.observation)

    @overrides
    def judge(self, model, *args, **kwargs):
        """
        Add optional model optimization functionality to :py:method:`sciunit.Test.judge` method, and delegate the rest
        of the work to the superclass.
        """
        if self.optimize_models:
            try:
                self.optimize(model)
            except Exception as e:
                logger().error(
                    f"{self.name} : Optimization procedure for model {model.name} has failed! Exception {e}"
                )

        return super().judge(model, *args, **kwargs)

    @overrides
    def optimize(self, model):
        """
        Optimize the given single- or multi-subject model using the fitting part of `self.observation` variable.
        If model is a single-subject model, its `fit` method will be called using the key-value pairs generated from
        `self.observation` dictionary.

        If model is a multi-subject model, its `fit_jointly` method will be called using key-value pairs generated
        from `self.observation` list (of dictionaries). As an example, if `self.observation = [{'k0': v00, 'k1': v01}, {'k0': v10, 'k1': v11}]`,
        then `fit_jointly` will be called as:

            `fit_jointly(k0=[v00, v10], k1=[v01, v11])`
        """
        obs = self.get_fitting_observations()
        logger().info(f"{self.name} : Optimizing {model.name} model...")
        if self.multi_subject:
            dict_of_lists = defaultdict(list)
            for subj_obs in obs:
                for k, v in subj_obs.items():
                    dict_of_lists[k].append(v)
            model.fit_jointly(**dict_of_lists)
        else:
            model.fit(**obs)

    @overrides
    def generate_prediction(self, model):
        """
        Given a multi or single subject model, run the tests to generate predictions and return them. If the test is
        a multi-subject test, the model must be derived from `ldmunit.capabilities.MultiSubjectModel`.

        See Also
        --------
        :py:function:`ldmunit.models.utils.multi_from_single_cls`, :class:`ldmunit.capabilities.MultiSubjectModel`
        """
        logger().debug(f"{self.name} : Generating predictions from {model.name}...")
        observations = self.get_testing_observations()
        if self.multi_subject:
            assert isinstance(
                model, MultiSubjectModel
            ), "Multi subject tests can only accept multi subject models"

            n_subj = len(observations)
            predictions = []
            score_kwargs = []
            for subj_idx in range(n_subj):
                single_subj_adapter = single_from_multi_obj(model, subj_idx)
                try:
                    pred_single = self.predict_single(
                        single_subj_adapter, observations[subj_idx]
                    )
                except Exception as e:
                    logger().error(
                        f"{self.name} : {model.name} predict_single call has failed! Exception: {e}"
                    )
                    pred_single = []

                predictions.append(pred_single)
                score_kwargs.append(
                    self.get_kwargs_for_compute_score(
                        model, observations[subj_idx], pred_single
                    )
                )
                model = reverse_single_from_multi_obj(single_subj_adapter)
        else:
            try:
                predictions = self.predict_single(model, observations)
            except Exception as e:
                logger().error(
                    f"{self.name} : {model.name} predict_single call has failed! Exception {e}"
                )
                predictions = []

            score_kwargs = self.get_kwargs_for_compute_score(
                model, observations, predictions
            )

        self.score_kwargs = score_kwargs
        return predictions

    def get_kwargs_for_compute_score(self, model, observations, predictions):
        if self.fn_kwargs_for_score is not None:
            return self.fn_kwargs_for_score(model, observations, predictions)
        else:
            return dict()

    @overrides
    def compute_score(self, _, predictions, **kwargs):
        """
        Compute the score from the given predictions and `self.observation`.
        """
        observations = self.get_testing_observations()
        if self.multi_subject:
            n_subj = len(observations)
            scores = []
            for subj_idx in range(n_subj):
                try:
                    single_score = self.compute_score_single(
                        observations[subj_idx],
                        predictions[subj_idx],
                        **self.score_kwargs[subj_idx],
                        **kwargs,
                    ).score
                except Exception as e:
                    logger().error(
                        f"{self.name} : compute_score_single has failed! Exception {e}"
                    )
                    single_score = np.NaN
                scores.append(single_score)
            score = self.score_type(self.score_aggr_fn(scores))
        else:
            try:
                score = self.compute_score_single(
                    observations, predictions, **self.score_kwargs, **kwargs
                )
            except Exception as e:
                logger().error(
                    f"{self.name} : compute_score_single has failed! Exception {e}"
                )
                score = self.score_type(np.NaN)
        return score

    def predict_single(self, model, observations, **kwargs):
        """
        Generate predictions for one group of testing. In the single subject case, this is the main prediction
        generation method, given the model and observations. In the multi subject case, each call to this method
        contains one of the single-subject models and the corresponding observations for that subject. In either case,
        users don't need to handle multi-subject prediction handling themselves, and can just implement the single-subject
        case.

        Parameters
        ----------
        model : :class:`ldmunit.models.LDMModel`
            A single-subject model. Method calls don't need to pass any subject index, even if the model is multi-subject.

        observations : dict
            Single-subject observation dictionary containing the keys specific to the concrete test class.

        Returns
        -------
        predictions : array-like
            Predictions for each of the stimuli in `observations` dictionary, in the same order.
        """
        raise NotImplementedError(
            "predict_single must be implemented by concrete Test classes"
        )

    def compute_score_single(self, observations, predictions, **kwargs):
        """
        Compute the score for one group of testing. Similar to `predict_single`, in the single subject case, this is the
        main score computing method, given observations, predictions and possible extra keyword arguments generated by `fn_kwargs_for_score`.
        In the multi-subject case, all the inputs are given for each subject with separate calls to this function. In either case,
        users don't need to handle multi-subject case themselves, and can just implement the single-subject case.

        Parameters
        ----------
        observations : dict
            Observations dictionary containing the keys specific to the concrete test class.

        predictions : array-like
            Predictions for each of the stimuli in `observations` dictionary, in the same order.

        Other Parameters
        ----------------
        kwargs : dict
            Optional keyword arguments required for the particular score being computed.

        Returns
        -------
        score : `sciunit.Score`
            The score object.
        """
        raise NotImplementedError(
            "compute_score_single must be implemented by concrete Test classes"
        )

    @overrides
    def bind_score(self, score, model, observation, prediction):
        """
        Override parent `bind_score` method to allow (optional) persistence.
        """
        self.persist(score, model, prediction)

    def persist(self, score, model, prediction):
        """
        Persist the results of the test if a path is given.

        Parameters
        ----------
        score : :class:`sciunit.Score`
            Final score object

        model : :class:`ldmunit.models.LDMModel`
            Tested model. In order to save the model, the model should implement `save(output_path)` method.

        prediction : dict or sequence of dict
            Predictions generated during the test
        """
        if self.persist_path is None:
            return

        folderpath = pathjoin(self.persist_path, model.name)
        makedirs(folderpath, exist_ok=True)
        score_filepath = pathjoin(folderpath, "score")
        pred_filepath = pathjoin(folderpath, "predictions")
        model_filepath = pathjoin(folderpath, "model")
        try:
            self.persist_score(score_filepath, score)
        except Exception as e:
            logger().error(f"{self.name} : persist_score has failed! Exception {e}")
        try:
            self.persist_predictions(pred_filepath, prediction)
        except Exception as e:
            logger().error(
                f"{self.name} : persist_predictions has failed! Exception {e}"
            )
        try:
            self.persist_model(model_filepath, model)
        except Exception as e:
            logger().error(f"{self.name} : persist_model has failed! Exception {e}")

        logger().info("Test results have been persisted")

    def persist_score(self, path, score):
        """
        Persist the score value in the given path.
        """
        np.save(path, np.asarray(score.score))
        logger().debug(f"Score is saved in {path}")

    def persist_predictions(self, path, predictions):
        """
        Persist the predictions in the given path.
        """
        np.save(path, np.asarray(predictions))
        logger().debug(f"Predictions are saved in {path}")

    def persist_model(self, path, model):
        """
        Persist the model in the given path, if `model` implements `save(path)` method.
        """
        try:
            model.save(path)
            logger().debug(f"Model is saved in {path}")
        except AttributeError:
            logger().debug(
                f"Model {model.name} does not implement save method; model has not been saved."
            )
