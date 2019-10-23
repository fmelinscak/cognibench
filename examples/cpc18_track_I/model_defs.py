import pandas as pd
import numpy as np
import time
from ldmunit.testing.tests import MSETest
from ldmunit.models import CACO
from ldmunit.capabilities import Interactive
from sciunit import TestSuite
from sciunit import settings as sciunit_settings

import importlib


class BEASTsdPython(CACO):
    name = "BEASTsdPython"

    def __init__(self, *args, import_base_path, **kwargs):
        import_file = f"{import_base_path}.CPC18_BEASTsd_pred"
        self.module = importlib.import_module(import_file)
        super().__init__(*args, **kwargs)

    def predict(self, stimulus):
        return self.module.CPC18_BEASTsd_pred(*stimulus)

    def act(self, stimulus):
        return self.predict(stimulus)


class BEASTsdMATLAB(CACO):
    name = "BEASTsdMATLAB"

    def __init__(self, *args, import_base_path, **kwargs):
        pass

    def predict(self, stimulus):
        pass

    def act(self, stimulus):
        return self.predict(stimulus)


class BEASTsdR(CACO):
    name = "BEASTsdR"

    def __init__(self, *args, import_base_path, **kwargs):
        pass

    def predict(self, stimulus):
        pass

    def act(self, stimulus):
        return self.predict(stimulus)
