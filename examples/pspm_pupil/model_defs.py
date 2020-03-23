from cognibench.models import CNBModel
from cognibench.capabilities import ContinuousAction, ContinuousObservation
from cognibench.continuous import ContinuousSpace
from cognibench.models.wrappers import MatlabWrapperMixin


class PsPMModel(MatlabWrapperMixin, CNBModel, ContinuousAction, ContinuousObservation):
    name = "PsPM model"

    def __init__(
        self, *args, lib_paths, import_base_path, predict_fn, model_spec, **kwargs
    ):
        self.set_action_space(ContinuousSpace())
        self.set_observation_space(ContinuousSpace())

        def pred(matlab_sess, stimuli):
            stimuli_copy = dict(stimuli)
            stimuli_copy.update(model_spec)
            return matlab_sess.feval(predict_fn, stimuli_copy)

        MatlabWrapperMixin.__init__(
            self,
            lib_paths=lib_paths,
            import_base_path=import_base_path,
            predict_fn=pred,
        )
        CNBModel.__init__(self, *args, **kwargs)
