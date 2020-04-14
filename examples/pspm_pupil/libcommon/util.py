from collections import namedtuple
from cognibench.utils import partialclass
import cognibench.scores as scores
from cognibench.testing.tests import BatchTest
from sciunit import TestSuite
from shutil import copytree, rmtree
from os.path import exists, join as pathjoin
import numpy as np

Dataset = namedtuple("Dataset", "name subject_ids")

DATA_PATH = "data"
OUT_PATH = "output"
LIB_PATHS = ["/home/eozd/bachlab/pspm/src", "libcommon"]


class Dataset:
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


def temp_dataset(tmp_base_path, dataset):
    dest_path = pathjoin(tmp_base_path, dataset.name)
    if exists(dest_path):
        rmtree(dest_path)
    copytree(dataset.path, dest_path)
    return dest_path


def get_obs_dict_list(exp_output_path):
    obs_dict_list = [
        {
            "stimuli": {
                "datapath": temp_dataset(exp_output_path, ds),
                "subject_ids": ds.subject_ids,
            },
            "actions": [],
        }
        for ds in DATASET_LIST
    ]
    return obs_dict_list


def get_test_suite(exp_output_path):
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
