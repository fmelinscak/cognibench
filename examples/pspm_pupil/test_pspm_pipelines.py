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
    obs_dict0 = {"stimuli": pathjoin(DATA_PATH, "PIT2e1.asc"), "actions": actions}
    obs_dict1 = {"stimuli": pathjoin(DATA_PATH, "PIT2e2.asc"), "actions": actions}
    obs_dict2 = {"stimuli": pathjoin(DATA_PATH, "PIT2e3.asc"), "actions": actions}
    obs_dict3 = {"stimuli": pathjoin(DATA_PATH, "PIT2e38.asc"), "actions": actions}

    # prepare models
    models = [
        PsPMModel(
            pspm_path=PSPM_PATH,
            import_base_path=MODEL_PATH,
            predict_fn=f"model{i}",
            name=f"model{i}",
        )
        for i in range(6)
    ]

    # prepare tests
    MSEScore = partialclass(scores.MSEScore, min_score=0, max_score=1)
    persist_path_fmt = "output/{}"
    suite = TestSuite(
        [
            BatchTest(
                name=f"{fname} Test",
                observation={"stimuli": pathjoin(DATA_PATH, fname), "actions": actions},
                score_type=MSEScore,
                persist_path=persist_path_fmt.format(fname),
                optimize_models=False,
                logging=2,
            )
            for fname in ["PIT2e1.asc", "PIT2e2.asc", "PIT2e3.asc", "PIT2e38.asc"]
        ],
        name="Batch test suite",
    )

    # judge
    sm = suite.judge(models)
    print(sm)
