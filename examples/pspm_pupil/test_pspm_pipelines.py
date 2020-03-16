from os import getcwd
from collections import namedtuple
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

Dataset = namedtuple("Dataset", "name subject_ids")
dataset_list = [
    Dataset(name="sc4b", subject_ids=np.arange(1, 22)),
    Dataset(name="fss6b", subject_ids=np.arange(1, 19)),
    Dataset(name="vc7b", subject_ids=np.arange(1, 22)),
]
obs_dict_list = [
    {
        "stimuli": {
            "datapath": pathjoin(DATA_PATH, ds.name),
            "subject_ids": ds.subject_ids,
        },
        "actions": [],
    }
    for ds in dataset_list
]

if __name__ == "__main__":
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
                name=f"{ds.name} Test",
                observation=obs_dict,
                score_type=CohensD,
                persist_path=persist_path_fmt.format(ds.name),
                optimize_models=False,
                logging=2,
            )
            for obs_dict, ds in zip(obs_dict_list, dataset_list)
        ],
        name="Batch test suite",
    )

    # judge
    sm = suite.judge(models)
    print(sm)
