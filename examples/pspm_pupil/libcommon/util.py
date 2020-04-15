from collections import namedtuple
from cognibench.utils import partialclass
import cognibench.scores as scores
from cognibench.testing.tests import BatchTest
from sciunit import TestSuite
from shutil import copytree, rmtree
from os.path import exists, join as pathjoin
import numpy as np
import pandas as pd

DATA_PATH = "data"
OUT_PATH = "output"
LIB_PATHS = ["/home/eozd/bachlab/pspm/src", "libcommon"]


class Dataset:
    """
    Simple class to represent a dataset (e.g. doxmem2)
    """

    def __init__(self, *args, name, subject_ids):
        self.name = name
        self.subject_ids = subject_ids
        self.path = pathjoin(DATA_PATH, name)


DATASET_LIST = [
    Dataset(name="doxmem2", subject_ids=np.arange(1, 80)),
    Dataset(name="fer02", subject_ids=np.arange(1, 75)),
    Dataset(name="fss6b", subject_ids=np.arange(1, 19)),
    Dataset(name="li", subject_ids=np.arange(1, 21)),
    Dataset(name="pubfe", subject_ids=np.arange(1, 23)),
    Dataset(name="sc4b", subject_ids=np.arange(1, 22)),
    Dataset(name="vc7b", subject_ids=np.arange(1, 22)),
]


def sm_to_pandas(sm):
    """
    Convert a score matrix that is the result of sciunit TestSuite
    to a regular pandas dataframe.
    """
    n_rows, n_cols = sm.shape
    arr = np.empty(shape=(n_rows, n_cols), dtype=np.float64)
    for i in range(n_rows):
        for j in range(n_cols):
            arr[i, j] = sm.iat[i, j].score
    df = pd.DataFrame(data=arr, index=sm.index, columns=sm.columns)
    return df


def get_obs_dict_list(exp_output_path):
    """
    Get the observation dictionary that can be used in cognibench framework.
    """
    obs_dict_list = [
        {
            "stimuli": {
                "datapath": ds.path,
                "subject_ids": ds.subject_ids,
                "tmp_out_path": exp_output_path,
            },
            "actions": [],
        }
        for ds in DATASET_LIST
    ]
    return obs_dict_list


def get_test_suite(exp_output_path):
    """
    Construct a test suite for pupil benchmarking. Each dataset in DATASET_LIST
    will be a separate test in the suite.
    """
    # prepare tests
    obs_dict_list = get_obs_dict_list(exp_output_path)
    CohensD = partialclass(scores.CohensDScore, min_score=-5, max_score=5)
    suite = TestSuite(
        [
            BatchTest(
                name=f"{ds.name} Test",
                observation=obs_dict,
                score_type=CohensD,
                persist_path=pathjoin(exp_output_path, "log", ds.name),
                optimize_models=False,
                logging=2,
            )
            for obs_dict, ds in zip(obs_dict_list, DATASET_LIST)
        ],
        name="Batch test suite",
    )
    return suite
