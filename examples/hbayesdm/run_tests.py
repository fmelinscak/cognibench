import numpy as np
import sciunit
import pandas as pd
from os import getcwd
from os.path import join as pathjoin
from pathlib import Path

from cognibench.testing import BatchTest
from cognibench.utils import partialclass
import cognibench.scores as scores
from model_defs import HbayesdmModel

import hbayesdm.models as Hmodels

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
    df = pd.read_csv("data/bandit4arm_exampleData.txt", delimiter="\t")
    obs = dict()
    cols = ["subjID", "gain", "loss", "choice"]
    obs["stimuli"] = df[cols].values
    obs["actions"] = df["choice"].values
    n_data = len(obs["actions"])

    train_indices = np.arange(n_data)
    test_indices = train_indices
    suite = sciunit.TestSuite(
        [
            BatchTest(
                name="NLL Test",
                observation=obs,
                optimize_models=True,
                score_type=NLLScore,
                persist_path=Path("logs") / "nll",
                logging=2,
            ),
            BatchTest(
                name="AIC Test",
                observation=obs,
                optimize_models=True,
                score_type=AICScore,
                fn_kwargs_for_score=aic_kwargs_fn,
                persist_path=Path("logs") / "aic",
                logging=2,
            ),
            BatchTest(
                name="BIC Test",
                observation=obs,
                optimize_models=True,
                score_type=BICScore,
                fn_kwargs_for_score=bic_kwargs_fn,
                persist_path=Path("logs") / "bic",
                logging=2,
            ),
        ],
        name="4-Armed Bandit Task Suite",
    )

    model_names_fns = [
        ("2par lapse", Hmodels.bandit4arm_2par_lapse),
        ("4par", Hmodels.bandit4arm_4par),
        ("4par lapse", Hmodels.bandit4arm_lapse),
        ("Lapse decay", Hmodels.bandit4arm_lapse_decay),
    ]
    models = [
        HbayesdmModel(
            name=model_name,
            hbayesdm_model_func=model_fn,
            n_obs=4,
            n_action=4,
            col_names=cols,
            niter=50,
            nwarmup=25,
            nchain=4,
            ncore=4,
            seed=42,
        )
        for model_name, model_fn in model_names_fns
    ]
    suite.judge(models)


if __name__ == "__main__":
    main()
