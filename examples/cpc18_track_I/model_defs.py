from importlib import import_module
from cognibench.models import CNBModel
from cognibench.capabilities import ContinuousAction, ContinuousObservation
from cognibench.continuous import ContinuousSpace
from cognibench.models.wrappers import (
    OctaveWrapperMixin,
    RWrapperMixin,
)


class BEASTsdPython(CNBModel, ContinuousAction, ContinuousObservation):
    name = "BEASTsdPython"

    def __init__(self, *args, import_base_path, **kwargs):
        self.set_action_space(ContinuousSpace())
        self.set_observation_space(ContinuousSpace())
        fn_name = "CPC18_BEASTsd_pred"
        module = import_module(f"{import_base_path}.{fn_name}")
        self.pred_fn = getattr(module, fn_name)
        super().__init__(*args, **kwargs)

    def predict(self, *args, **kwargs):
        return self.pred_fn(*args, **kwargs)


class BEASTsdOctave(
    OctaveWrapperMixin, CNBModel, ContinuousAction, ContinuousObservation
):
    name = "BEASTsdOctave"

    def __init__(self, *args, import_base_path, **kwargs):
        self.set_action_space(ContinuousSpace())
        self.set_observation_space(ContinuousSpace())
        OctaveWrapperMixin.__init__(
            self,
            import_base_path=import_base_path,
            reset_fn=lambda oct_sess: oct_sess.eval("pkg load statistics;"),
            predict_fn="CPC18_BEASTsd_pred",
        )
        CNBModel.__init__(self, *args, **kwargs)
        self.reset()


class BEASTsdR(RWrapperMixin, CNBModel, ContinuousAction, ContinuousObservation):
    name = "BEASTsdR"

    def __init__(self, *args, import_base_path, **kwargs):
        self.set_action_space(ContinuousSpace())
        self.set_observation_space(ContinuousSpace())
        RWrapperMixin.__init__(
            self, import_base_path=import_base_path, predict_fn="CPC18_BEASTsd_pred"
        )
        CNBModel.__init__(self, *args, **kwargs)
