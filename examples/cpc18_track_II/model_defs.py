import pandas as pd
import numpy as np
from importlib import import_module
from ldmunit.models import CACO
from ldmunit.models.wrappers import RWrapperMixin
from overrides import overrides


class PythonModel(CACO):
    name = "PythonModel"

    @overrides
    def __init__(self, *args, import_base_path, **kwargs):
        self.module = import_module(f"{import_base_path}.model")
        super().__init__(*args, **kwargs)

    @overrides
    def reset(self):
        self.model = self.module.Model()

    @overrides
    def fit(self, stimuli, actions):
        return self.model.fit(stimuli, actions)

    @overrides
    def predict(self, stimuli):
        return self.model.predict(stimuli)


class RModel(RWrapperMixin, CACO):
    name = "RModel"

    @overrides
    def __init__(self, *args, import_base_path, **kwargs):
        RWrapperMixin.__init__(
            self,
            import_base_path=import_base_path,
            reset_fn="init",
            fit_fn="fit",
            predict_fn="predict_R",
        )
        CACO.__init__(self, *args, **kwargs)
