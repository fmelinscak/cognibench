import numpy as np
import sciunit
import pandas as pd
from os import getcwd
from os.path import join as pathjoin

from ldmunit.testing import BatchTrainAndTest
from ldmunit.utils import partialclass
import ldmunit.scores as scores
from model_defs import HbayesdmModel

from hbayesdm.models import bandit2arm_delta

DATA_PATH = "data"
# sciunit CWD directory should contain config.json file
sciunit.settings["CWD"] = getcwd()


NLLScore = partialclass(scores.NLLScore, min_score=0, max_score=1e4)
AICScore = partialclass(scores.AICScore, min_score=0, max_score=1e4)
BICScore = partialclass(scores.BICScore, min_score=0, max_score=1e4)


def aic_kwargs_fn(model, obs, pred):
    return {"n_model_params": model.n_params()}


def bic_kwargs_fn(model, obs, pred):
    return {"n_model_params": model.n_params(), "n_samples": len(obs["stimuli"])}


def main():
    df = pd.read_csv("data/bandit2arm_exampleData.txt", delimiter="\t")
    obs = dict()
    cols = ["subjID", "choice", "outcome"]
    obs["stimuli"] = df[cols].values
    obs["actions"] = df["choice"].values
    n_data = len(obs["actions"])

    train_indices = np.arange(n_data)
    test_indices = train_indices
    suite = sciunit.TestSuite(
        [
            BatchTrainAndTest(
                name="NLL Test",
                observation=obs,
                train_indices=train_indices,
                test_indices=test_indices,
                score_type=NLLScore,
                persist_path="logs",
                logging=2,
            ),
            BatchTrainAndTest(
                name="AIC Test",
                observation=obs,
                train_indices=train_indices,
                test_indices=test_indices,
                score_type=AICScore,
                fn_kwargs_for_score=aic_kwargs_fn,
                persist_path="logs",
                logging=2,
            ),
            BatchTrainAndTest(
                name="BIC Test",
                observation=obs,
                train_indices=train_indices,
                test_indices=test_indices,
                score_type=BICScore,
                fn_kwargs_for_score=bic_kwargs_fn,
                persist_path="logs",
                logging=2,
            ),
        ],
        name="2-Armed Bandit Task Suite",
    )

    models = [
        HbayesdmModel(
            name="Rescorla Wagner (hBayesDM)",
            hbayesdm_model_func=bandit2arm_delta,
            n_obs=2,
            n_action=2,
            col_names=cols,
            niter=500,
            nwarmup=250,
            nchain=4,
            ncore=4,
            seed=42,
        )
    ]
    suite.judge(models)


if __name__ == "__main__":
    main()
