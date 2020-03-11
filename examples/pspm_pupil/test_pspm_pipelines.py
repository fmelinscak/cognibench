from os import getcwd
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
    actions = [0]
    obs_dict = {"stimuli": pathjoin(DATA_PATH, "PIT2e1.asc"), "actions": actions}

    # prepare models
    models = [
        PsPMModel(
            pspm_path=PSPM_PATH, import_base_path=MODEL_PATH, predict_fn=f"model{i}"
        )
        for i in range(6)
    ]

    # prepare tests
    MSEScore = partialclass(scores.MSEScore, min_score=0, max_score=1)
    persist_path_fmt = "output/{}"
    suite = TestSuite(
        [
            BatchTest(
                name="MSE Test",
                observation=obs_dict,
                score_type=MSEScore,
                persist_path=persist_path_fmt.format("PIT2e1_test"),
                logging=2,
            ),
        ],
        name="Batch test suite",
    )

    # judge
    suite.judge(models)
