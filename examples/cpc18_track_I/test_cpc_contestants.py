from os import getcwd
import pandas as pd
import numpy as np
from ldmunit.testing.tests import BatchTest
from ldmunit.utils import partialclass
import ldmunit.scores as scores
from sciunit import TestSuite
from sciunit import settings as sciunit_settings

from model_defs import BEASTsdPython, BEASTsdOctave, BEASTsdR

sciunit_settings["CWD"] = getcwd()


def get_models(model_IDs, folder_name_fmt, model_name_fmt, model_ctor):
    models = []
    for model_id in model_IDs:
        folder_name = folder_name_fmt.format(model_id)
        model_name = model_name_fmt.format(model_id)
        models.append(model_ctor(import_base_path=folder_name, name=model_name))
    return models


def convert_to_numeric(df):
    for col in ["LotShapeA", "LotShapeB"]:
        arr = df[col].values
        _, inv = np.unique(arr, return_inverse=True)
        df[col] = inv
        df[col] = df[col].astype("int")
    return df


if __name__ == "__main__":
    # prepare data
    Data = pd.read_csv("CPC18_EstSet.csv")
    Data = convert_to_numeric(Data)
    stimuli = Data[
        [
            "Ha",
            "pHa",
            "La",
            "LotShapeA",
            "LotNumA",
            "Hb",
            "pHb",
            "Lb",
            "LotShapeB",
            "LotNumB",
            "Amb",
            "Corr",
        ]
    ].values
    actions = Data[["B.1", "B.2", "B.3", "B.4", "B.5"]].values
    obs_dict = {"stimuli": stimuli, "actions": actions}

    # prepare models
    python_model_IDs = [0]
    octave_model_IDs = [1]
    r_model_IDs = [2]
    models = (
        get_models(
            python_model_IDs,
            "beastsd_contestant_{}",
            "Contestant {} (Python)",
            BEASTsdPython,
        )
        + get_models(
            r_model_IDs, "beastsd_contestant_{}", "Contestant {} (R)", BEASTsdR
        )
        + get_models(
            octave_model_IDs,
            "beastsd_contestant_{}",
            "Contestant {} (Octave)",
            BEASTsdOctave,
        )
    )

    # prepare tests
    MSEScore = partialclass(scores.MSEScore, min_score=0, max_score=1)
    MAEScore = partialclass(scores.MAEScore, min_score=0, max_score=1)
    CrossEntropyScore = partialclass(
        scores.CrossEntropyScore, min_score=0, max_score=1000
    )
    persist_path_fmt = "output/{}"
    suite = TestSuite(
        [
            BatchTest(
                name="MSE Test",
                observation=obs_dict,
                score_type=MSEScore,
                persist_path=persist_path_fmt.format("mse_test"),
                logging=2,
            ),
            BatchTest(
                name="MAE Test",
                observation=obs_dict,
                score_type=MAEScore,
                persist_path=persist_path_fmt.format("mae_test"),
                logging=2,
            ),
            BatchTest(
                name="Cross Entropy Test",
                observation=obs_dict,
                score_type=CrossEntropyScore,
                persist_path=persist_path_fmt.format("cross_entropy_test"),
                logging=2,
            ),
        ],
        name="Batch test suite",
    )

    # judge
    suite.judge(models)

    # TODO: add ability to save testing output results for logging
    # np.savetxt("outputAll.csv", PredictedAll, delimiter=",", header = "B1,B2,B3,B4,B5", fmt='%.4f')
