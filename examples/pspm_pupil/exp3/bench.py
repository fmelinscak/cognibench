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
import sys
import os

sys.path.insert(0, os.getcwd())
from model_defs import PsPMModel

settings["CRASH_EARLY"] = True
sciunit_settings["CWD"] = getcwd()

DATA_PATH = "data"
MODEL_PATH = "exp3"
LIB_PATHS = ["/home/eozd/bachlab/pspm/src", "libcommon"]

Dataset = namedtuple("Dataset", "name subject_ids")
dataset_list = [
    Dataset(name="li", subject_ids=np.arange(1, 21)),
    Dataset(name="doxmem2", subject_ids=np.arange(1, 80)),
    Dataset(name="fss6b", subject_ids=np.arange(1, 19)),
    Dataset(name="pubfe", subject_ids=np.arange(1, 23)),
    Dataset(name="sc4b", subject_ids=np.arange(1, 22)),
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
    miss_percs = [10, 20, 35, 50, 60]
    models = [
        PsPMModel(
            lib_paths=LIB_PATHS,
            import_base_path=MODEL_PATH,
            predict_fn="fit_all",
            model_spec={"miss_perc_threshold": perc},
            name=f"Miss perc {perc}",
        )
        for perc in miss_percs
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
