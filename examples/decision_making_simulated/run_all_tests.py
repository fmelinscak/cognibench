import numpy as np
import sciunit
import pandas as pd
from os import getcwd
from os.path import join as pathjoin

from ldmunit.models import decision_making
from ldmunit.models.utils import multi_from_single_cls as multi_subject
from ldmunit.testing import InteractiveTest
from ldmunit.utils import partialclass
import ldmunit.scores as scores
from read_example_data import get_simulation_data, get_model_params

DATA_PATH = "data"
# sciunit CWD directory should contain config.json file
sciunit.settings["CWD"] = getcwd()

NLLScore = partialclass(scores.NLLScore, min_score=0, max_score=1000)
AICScore = partialclass(scores.AICScore, min_score=0, max_score=1000)
BICScore = partialclass(scores.BICScore, min_score=0, max_score=1000)


def aic_kwargs_fn(model, obs, pred):
    return {"n_model_params": model.n_params()}


def bic_kwargs_fn(model, obs, pred):
    return {"n_model_params": model.n_params(), "n_samples": len(obs["stimuli"])}


def get_tests(score_name, score_type, score_kwargs_fn):
    names_paths = [
        ("ck", pathjoin(DATA_PATH, "multi-ck.csv")),
        ("rw", pathjoin(DATA_PATH, "multi-rw.csv")),
        ("rwck", pathjoin(DATA_PATH, "multi-rwck.csv")),
        ("rr", pathjoin(DATA_PATH, "multi-random-responding.csv")),
        ("nwsls", pathjoin(DATA_PATH, "multi-nwsls.csv")),
    ]
    tests = []
    for test_name, path in names_paths:
        obs = get_simulation_data(path, 3, True)
        curr_test = InteractiveTest(
            multi_subject=True,
            name=f"{test_name} sim {score_name}",
            observation=obs,
            score_type=score_type,
            fn_kwargs_for_score=score_kwargs_fn,
        )
        tests.append(curr_test)
    return tests


def get_models():
    MultiRWModel = multi_subject(decision_making.RWModel)
    MultiCKModel = multi_subject(decision_making.CKModel)
    MultiRWCKModel = multi_subject(decision_making.RWCKModel)
    MultiNWSLSModel = multi_subject(decision_making.NWSLSModel)
    n_action, n_obs = 3, 3

    multi_ck = MultiCKModel(n_subj=3, n_action=n_action, n_obs=n_obs, seed=42)
    multi_ck.name = "ck"

    multi_rw = MultiRWModel(n_subj=3, n_action=n_action, n_obs=n_obs, seed=42)
    multi_rw.name = "rw"

    multi_rwck = MultiRWCKModel(n_subj=3, n_action=n_action, n_obs=n_obs, seed=42)
    multi_rwck.name = "rwck"

    multi_nwsls = MultiNWSLSModel(n_subj=3, n_action=n_action, n_obs=n_obs, seed=42)
    multi_nwsls.name = "nwsls"

    return [multi_ck, multi_rw, multi_rwck, multi_nwsls]


def main():
    nll_al_suite = sciunit.TestSuite(
        get_tests("NLL", NLLScore, None), name="NLL suite for decision making"
    )
    aic_al_suite = sciunit.TestSuite(
        get_tests("AIC", AICScore, aic_kwargs_fn), name="AIC suite for decision making"
    )
    bic_al_suite = sciunit.TestSuite(
        get_tests("BIC", BICScore, bic_kwargs_fn), name="BIC suite for decision making"
    )

    models = get_models()
    # Run test suites
    # -----------------------------------------------
    nll_al_suite.judge(models)
    aic_al_suite.judge(models)
    bic_al_suite.judge(models)


if __name__ == "__main__":
    main()
