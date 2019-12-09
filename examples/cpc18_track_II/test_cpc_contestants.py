from os import getcwd
import pandas as pd
import numpy as np
import time
from ldmunit.testing import BatchTestWithSplit
import ldmunit.scores as scores
from ldmunit.utils import partialclass
from ldmunit.models import CACO
from ldmunit.capabilities import Interactive
from sciunit import TestSuite
from sciunit import settings as sciunit_settings

from model_defs import PythonModel, RModel

sciunit_settings["CWD"] = getcwd()


def get_models(model_IDs, folder_name_fmt, model_name_fmt, model_ctor):
    models = []
    for model_id in model_IDs:
        folder_name = folder_name_fmt.format(model_id)
        model_name = model_name_fmt.format(model_id)
        models.append(model_ctor(import_base_path=folder_name, name=model_name))
    return models


# Transform outputs to be consistent with maximization (i.e., not minimization)
def pBpMaxTransform(orig_vec, is_b_max):
    orig_vec.name = "B"
    new_vec = pd.concat([orig_vec, is_b_max], axis=1)
    new_vec.loc[new_vec["isBMax"] == False, "B"] = 1 - new_vec["B"]
    return new_vec["B"]


def getSplit(dd, seed=1, nSubjTest=30, nGamesPerSubjTest=5):

    if "SubjID" not in dd.columns or "GameID" not in dd.columns:
        print("data must include SubjID and GameID columns")
        return None, None

    dd = dd.sort_values(by=["SubjID", "GameID"])
    np.random.seed(seed)
    subjs = np.array(list(dd["SubjID"].unique()))
    subj2remove = np.random.choice(subjs, nSubjTest, replace=False)
    train = np.ones(dd.shape[0], dtype=np.bool)
    test = np.zeros(dd.shape[0], dtype=np.bool)
    for s in range(0, nSubjTest):
        subjD = dd.loc[dd["SubjID"] == subj2remove[s]]
        games = np.array(list(subjD["GameID"].unique()))
        games2remove = np.random.choice(games, nGamesPerSubjTest, replace=False)
        for g in range(0, nGamesPerSubjTest):
            mask = (dd["SubjID"] == subj2remove[s]) & (dd["GameID"] == games2remove[g])
            test |= mask
            train &= ~mask
    train_idx = np.where(train)[0]
    test_idx = np.where(test)[0]
    return train_idx, test_idx


def convert_to_numeric(df):
    for col in ["Location", "Gender", "LotShapeA", "LotShapeB"]:
        arr = df[col].values
        _, inv = np.unique(arr, return_inverse=True)
        df[col] = inv
        df[col] = df[col].astype("int")
    return df


if __name__ == "__main__":
    # prepare data
    df = pd.read_csv("individualBlockAvgs.csv")
    df = convert_to_numeric(df)
    df = df.drop(columns="BEAST_blkPred")

    is_b_max = df["diffEV"] >= 0
    is_b_max.name = "isBMax"
    df["B"] = pBpMaxTransform(df["B"], is_b_max)

    first_part = df.loc[df.SubjID < 60000]
    second_part = df.loc[df.SubjID >= 60000]
    train_indices, test_indices = getSplit(second_part, seed=1)
    obs_train = df.iloc[train_indices]
    obs_test = df.iloc[test_indices]

    train_stimuli = obs_train.values[:, :-1]
    train_actions = obs_train.values[:, -1].astype(np.float64)
    test_stimuli = obs_test.values[:, :-1]
    test_actions = obs_test.values[:, -1].astype(np.float64)
    obs_dict = {
        "train": {"stimuli": train_stimuli, "actions": train_actions},
        "test": {"stimuli": test_stimuli, "actions": test_actions},
    }

    # prepare models
    python_model_IDs = [0]
    r_model_IDs = [1, 2]
    models = get_models(
        python_model_IDs, "contestant_{}", "Contestant {} (Python)", PythonModel
    ) + get_models(r_model_IDs, "contestant_{}", "Contestant {} (R)", RModel)

    # prepare tests
    MSEScore = partialclass(scores.MSEScore, min_score=0, max_score=1)
    MAEScore = partialclass(scores.MAEScore, min_score=0, max_score=1)
    CrossEntropyScore = partialclass(
        scores.CrossEntropyScore, min_score=0, max_score=1000
    )
    PearsonCorrScore = scores.PearsonCorrelationScore
    persist_path_fmt = "output/{}"
    suite = TestSuite(
        [
            BatchTestWithSplit(
                name="MSE Test",
                observation=obs_dict,
                score_type=MSEScore,
                persist_path=persist_path_fmt.format("mse_test"),
                logging=2,
            ),
            BatchTestWithSplit(
                name="MAE Test",
                observation=obs_dict,
                score_type=MAEScore,
                persist_path=persist_path_fmt.format("mae_test"),
                logging=2,
            ),
            BatchTestWithSplit(
                name="Cross Entropy Test",
                observation=obs_dict,
                score_type=CrossEntropyScore,
                persist_path=persist_path_fmt.format("cross_entropy_test"),
                logging=2,
            ),
            BatchTestWithSplit(
                name="Pearson Correlation Test",
                observation=obs_dict,
                score_type=PearsonCorrScore,
                persist_path=persist_path_fmt.format("corr_test"),
                logging=2,
            ),
        ],
        name="Batch train and test suite",
    )

    # judge
    suite.judge(models)
