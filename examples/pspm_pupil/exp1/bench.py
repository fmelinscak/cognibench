from os import getcwd
import pandas as pd
import itertools
import numpy as np
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

MODEL_PATH = "exp1"
EXP_OUTPUT_PATH = pathjoin(util.OUT_PATH, MODEL_PATH)
SEGMENT_LENGTH_LIST = [2.5, 5, 7.5, 10]
CUTOFF_LIST = [30, 50, 70]

if __name__ == "__main__":
    os.makedirs(EXP_OUTPUT_PATH, exist_ok=True)
    models = [
        PsPMModel(
            lib_paths=util.LIB_PATHS,
            import_base_path=MODEL_PATH,
            predict_fn="fit_all",
            model_spec={"exclude_segment_length": seglen, "exclude_cutoff": cutoff},
            name=f"Seglen {seglen:2.1f} Cutoff {cutoff}",
        )
        for seglen, cutoff in itertools.product(SEGMENT_LENGTH_LIST, CUTOFF_LIST)
    ]
    suite = util.get_test_suite(EXP_OUTPUT_PATH)

    # judge
    sm = suite.judge(models)
    print(sm)
    sm_df = util.sm_to_pandas(sm)
    sm_df.to_csv(pathjoin(EXP_OUTPUT_PATH, "score_matrix.csv"), na_rep="NULL")
