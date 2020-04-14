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

MODEL_PATH = "exp4"
EXP_OUTPUT_PATH = pathjoin(util.OUT_PATH, MODEL_PATH)

if __name__ == "__main__":
    # prepare models
    model_list = [
        (
            "single-trial",
            {
                # TODO: enter optimal single-trial model
                "trial_exclusion_threshold": 20,
                "participant_exclusion_threshold": 35,
                "exclude_segment_length": 10,
            },
        ),
        # TODO: enter optimal condition-wise model
        ("condition-wise", {"participant_exclusion_threshold": 35}),
    ]
    models = [
        PsPMModel(
            lib_paths=util.LIB_PATHS,
            import_base_path=MODEL_PATH,
            predict_fn="fit_all",
            model_spec=dict({"model_str": model_str}, **params),
            name=model_str,
        )
        for model_str, params in model_list
    ]

    suite = util.get_test_suite(EXP_OUTPUT_PATH)

    # judge
    sm = suite.judge(models)
    print(sm)
    sm_df = util.sm_to_pandas(sm)
    sm_df.to_csv(pathjoin(EXP_OUTPUT_PATH, "score_matrix.csv"), na_rep="NULL")
