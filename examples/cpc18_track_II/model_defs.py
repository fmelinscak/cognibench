import pandas as pd
import numpy as np
from importlib import import_module
from ldmunit.models import LDMModel
from ldmunit.capabilities import ContinuousAction, ContinuousObservation
from ldmunit.continuous import ContinuousSpace
from ldmunit.models.wrappers import RWrapperMixin
from overrides import overrides


class PythonModel(LDMModel, ContinuousAction, ContinuousObservation):
    name = "PythonModel"

    @overrides
    def __init__(self, *args, import_base_path, **kwargs):
        self.set_action_space(ContinuousSpace())
        self.set_observation_space(ContinuousSpace())
        self.module = import_module(f"{import_base_path}.model")
        super().__init__(*args, **kwargs)
        self.reset()

    @overrides
    def reset(self):
        self.model = self.module.Model()

    @overrides
    def fit(self, stimuli, actions):
        return self.model.fit(stimuli, actions)

    @overrides
    def predict(self, stimuli):
        return self.model.predict(stimuli)


class RModel(RWrapperMixin, LDMModel, ContinuousAction, ContinuousObservation):
    name = "RModel"

    @overrides
    def __init__(self, *args, import_base_path, **kwargs):
        self.set_action_space(ContinuousSpace())
        self.set_observation_space(ContinuousSpace())
        RWrapperMixin.__init__(
            self,
            import_base_path=import_base_path,
            reset_fn="init",
            fit_fn="fit",
            predict_fn="predict_R",
        )
        LDMModel.__init__(self, *args, **kwargs)
        self.reset()
