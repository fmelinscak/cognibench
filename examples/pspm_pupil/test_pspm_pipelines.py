from os import getcwd
import numpy as np
from os.path import join as pathjoin
from cognibench.testing.tests import BatchTest
from cognibench.utils import partialclass
import cognibench.scores as scores
from sciunit import TestSuite
from sciunit import settings as sciunit_settings
from cognibench.settings import settings
from model_defs import PsPMModel

settings["CRASH_EARLY"] = True
sciunit_settings["CWD"] = getcwd()

DATA_PATH = "data"
MODEL_PATH = "pipelines"
PSPM_PATH = "/home/eozd/bachlab/pspm/src"

if __name__ == "__main__":
    # prepare data
    obs_dict0 = {
        "stimuli": {
            "datapath": pathjoin(DATA_PATH, "fss6b"),
            "subject_ids": np.arange(1, 19),
        },
        "actions": [],
    }
    # obs_dict1 = {"stimuli": pathjoin(DATA_PATH, "PIT2e2.asc"), "actions": actions}
    # obs_dict2 = {"stimuli": pathjoin(DATA_PATH, "PIT2e3.asc"), "actions": actions}
    # obs_dict3 = {"stimuli": pathjoin(DATA_PATH, "PIT2e38.asc"), "actions": actions}

    # prepare models
    model_names = [f"model{i}" for i in range(6)]
    models = [
        PsPMModel(
            pspm_path=PSPM_PATH,
            import_base_path=MODEL_PATH,
            predict_fn=f"{model_name}",
            name=f"{model_name}",
        )
        for model_name in model_names
    ]

    # prepare tests
    CohensD = partialclass(scores.CohensDScore, min_score=-5, max_score=5)
    persist_path_fmt = "output/{}"
    suite = TestSuite(
        [
            BatchTest(
                name="FSS6B Test",
                observation=obs_dict0,
                score_type=CohensD,
                persist_path=persist_path_fmt.format("FSS6B"),
                optimize_models=False,
                logging=2,
            )
        ],
        name="Batch test suite",
    )

    # judge
    sm = suite.judge(models)
    print(sm)
