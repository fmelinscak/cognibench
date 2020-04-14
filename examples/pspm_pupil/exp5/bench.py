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

MODEL_PATH = "exp5"
EXP_OUTPUT_PATH = pathjoin(util.OUT_PATH, MODEL_PATH)
DISCARD_FACTOR_LIST = [0.0, 0.005, 0.01, 0.015, 0.02]

if __name__ == "__main__":
    # prepare models
    model_list = [
        (f"blink_saccade {factor:.3f}", {"discard_factor": factor})
        for factor in DISCARD_FACTOR_LIST
    ] + [("pupil_pp", {})]
    # TODO: pupil preprocessing options
    models = [
        PsPMModel(
            lib_paths=util.LIB_PATHS,
            import_base_path=MODEL_PATH,
            predict_fn="fit_all",
            model_spec=dict({"model_str": model[0]}, **model[1]),
            name=model[0],
        )
        for model in model_list
    ]

    suite = util.get_test_suite(EXP_OUTPUT_PATH)

    # judge
    sm = suite.judge(models)
    print(sm)
