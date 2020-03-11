from cognibench.models import CNBModel
from cognibench.capabilities import ContinuousAction, ContinuousObservation
from cognibench.continuous import ContinuousSpace
from cognibench.models.wrappers import MatlabWrapperMixin


class PsPMModel(MatlabWrapperMixin, CNBModel, ContinuousAction, ContinuousObservation):
    name = "PsPM model"

    def __init__(self, *args, pspm_path, import_base_path, predict_fn, **kwargs):
        self.set_action_space(ContinuousSpace())
        self.set_observation_space(ContinuousSpace())
        MatlabWrapperMixin.__init__(
            self,
            pspm_path=pspm_path,
            import_base_path=import_base_path,
            predict_fn=predict_fn,
        )
        CNBModel.__init__(self, *args, **kwargs)
