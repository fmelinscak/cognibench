from os import getcwd
import pandas as pd
import numpy as np
import time
from ldmunit.testing.tests import MSETest
from ldmunit.models import CACO
from ldmunit.capabilities import Interactive
from sciunit import TestSuite
from sciunit import settings as sciunit_settings

from model_defs import BEASTsdPython, BEASTsdMATLAB, BEASTsdR

sciunit_settings["CWD"] = getcwd()


if __name__ == "__main__":
    # prepare data
    Data = pd.read_csv("CPC18_EstSet.csv")
    stimuli = Data[
        [
            "Ha",
            "pHa",
            "La",
            "LotShapeA",
            "LotNumA",
            "Hb",
            "pHb",
            "Lb",
            "LotShapeB",
            "LotNumB",
            "Amb",
            "Corr",
        ]
    ].values
    actions = Data[["B.1", "B.2", "B.3", "B.4", "B.5"]].values
    obs_dict = {"stimuli": stimuli, "actions": actions}

    # prepare models
    models = [
        BEASTsdPython(
            import_base_path=f"beastsd_contestant_{i}", name=f"Contestant {i}"
        )
        for i in range(3)
    ]

    # prepare tests
    test = MSETest(name="MSE Test", observation=obs_dict)
    suite = TestSuite([test], name="MSE suite")

    # judge
    suite.judge(models)

    # TODO: add ability to save testing output results for logging
    # np.savetxt("outputAll.csv", PredictedAll, delimiter=",", header = "B1,B2,B3,B4,B5", fmt='%.4f')
