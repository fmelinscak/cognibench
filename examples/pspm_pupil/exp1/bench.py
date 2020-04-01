from os import getcwd
import itertools
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
MODEL_PATH = "exp1"
LIB_PATHS = ["/home/eozd/bachlab/pspm/src", "libcommon"]

Dataset = namedtuple("Dataset", "name subject_ids")
dataset_list = [
    Dataset(name="doxmem2", subject_ids=np.arange(1, 80)),
    Dataset(name="fer02", subject_ids=np.arange(1, 75)),
    Dataset(name="fss6b", subject_ids=np.arange(1, 19)),
    Dataset(name="li", subject_ids=np.arange(1, 21)),
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
    segment_length_list = [2.5, 5, 7.5, 10]
    cutoff_list = [30, 50, 70]
    models = [
        PsPMModel(
            lib_paths=LIB_PATHS,
            import_base_path=MODEL_PATH,
            predict_fn="fit_all",
            model_spec={"exclude_segment_length": seglen, "exclude_cutoff": cutoff},
            name=f"Seglen {seglen:2.1f} Cutoff {cutoff}",
        )
        for seglen, cutoff in itertools.product(segment_length_list, cutoff_list)
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