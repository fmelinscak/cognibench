from os import getcwd
from os.path import join as pathjoin
from sciunit import settings as sciunit_settings
from cognibench.settings import settings
import sys
import os

sys.path.insert(0, os.getcwd())
from model_defs import PsPMModel
from libcommon import util

settings["CRASH_EARLY"] = True
sciunit_settings["CWD"] = getcwd()

MODEL_PATH = "exp6"
EXP_OUTPUT_PATH = pathjoin(util.OUT_PATH, MODEL_PATH)
ANGLE_LIST = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]

if __name__ == "__main__":
    # prepare models
    models = [
        PsPMModel(
            lib_paths=util.LIB_PATHS,
            import_base_path=MODEL_PATH,
            predict_fn="fit_all",
            model_spec={"fixation_angle": angle},
            name=f"Fixation angle: {angle}",
        )
        for angle in ANGLE_LIST
    ]

    suite = util.get_test_suite(EXP_OUTPUT_PATH)

    # judge
    sm = suite.judge(models)
    print(sm)
