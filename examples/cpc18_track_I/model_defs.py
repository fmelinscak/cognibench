import pandas as pd
import numpy as np
import time
from ldmunit.testing.tests import MSETest
from ldmunit.models import CACO
from ldmunit.capabilities import Interactive
from sciunit import TestSuite
from sciunit import settings as sciunit_settings
import os

from oct2py import Oct2Py
import rpy2.robjects as robj
from rpy2.robjects.packages import SignatureTranslatedAnonymousPackage as STAP
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


class BEASTsdOctave(CACO):
    name = "BEASTsdOctave"

    def __init__(self, *args, import_base_path, **kwargs):
        self.octave_session = Oct2Py()
        self.octave_session.eval("pkg load statistics")
        self.octave_session.eval(f'addpath("{import_base_path}");')
        super().__init__(*args, **kwargs)

    def predict(self, stimulus):
        return self.octave_session.feval("CPC18_BEASTsd_pred", *stimulus)

    def act(self, stimulus):
        return self.predict(stimulus)


class BEASTsdR(CACO):
    name = "BEASTsdR"

    def __init__(self, *args, import_base_path, **kwargs):
        r_files = [f for f in os.listdir(import_base_path) if f.lower().endswith(".r")]
        r_codestring = ""
        for filename in r_files:
            with open(os.path.join(import_base_path, filename), "r") as f:
                r_codestring += f.read()
                r_codestring += "\n"
        self.r_predict = STAP(r_codestring, "r_predict")
        super().__init__(*args, **kwargs)

    def predict(self, stimulus):
        return self.r_predict.CPC18_BEASTsd_pred(*stimulus)

    def act(self, stimulus):
        return self.predict(stimulus)
