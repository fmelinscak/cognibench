import numpy as np
import sciunit
import pandas as pd
from os import getcwd
from os.path import join as pathjoin

from cognibench.models import associative_learning
from cognibench.models.utils import multi_from_single_cls as multi_subject
from cognibench.testing import InteractiveTest
from cognibench.utils import partialclass
import cognibench.scores as scores
from read_example_data import get_simulation_data

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
        ("rr_al", pathjoin(DATA_PATH, "multi-rr_al.csv")),
        ("rw_norm", pathjoin(DATA_PATH, "multi-rw_norm.csv")),
        ("krw_norm", pathjoin(DATA_PATH, "multi-krw_norm.csv")),
        ("lsspd", pathjoin(DATA_PATH, "multi-lsspd.csv")),
        ("beta_binomial", pathjoin(DATA_PATH, "multi-beta_binomial.csv")),
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
            persist_path=pathjoin("results", f"{test_name}_{score_name}"),
        )
        tests.append(curr_test)
    return tests


def get_models():
    MultiRwNormModel = multi_subject(associative_learning.RwNormModel)
    MultiKrwNormModel = multi_subject(associative_learning.KrwNormModel)
    MultiBetaBinomialModel = multi_subject(associative_learning.BetaBinomialModel)
    MultiLSSPDModel = multi_subject(associative_learning.LSSPDModel)

    multi_rw_norm = MultiRwNormModel(n_subj=3, n_obs=4, seed=42)
    multi_rw_norm.name = "rw_norm"

    multi_krw_norm = MultiKrwNormModel(n_subj=3, n_obs=4, seed=42)
    multi_krw_norm.name = "krw_norm"

    multi_lsspd = MultiLSSPDModel(n_subj=3, n_obs=4, seed=42)
    multi_lsspd.name = "lsspd"

    multi_bb = MultiBetaBinomialModel(n_subj=3, n_obs=4, seed=42)
    multi_bb.name = "bb"

    return [multi_rw_norm, multi_krw_norm, multi_lsspd, multi_bb]


def main():
    nll_al_suite = sciunit.TestSuite(
        get_tests("NLL", NLLScore, None), name="NLL suite for associative learning"
    )
    aic_al_suite = sciunit.TestSuite(
        get_tests("AIC", AICScore, aic_kwargs_fn),
        name="AIC suite for associative learning",
    )
    bic_al_suite = sciunit.TestSuite(
        get_tests("BIC", BICScore, bic_kwargs_fn),
        name="BIC suite for associative learning",
    )

    models = get_models()
    # Run test suites
    # -----------------------------------------------
    nll_al_suite.judge(models)
    aic_al_suite.judge(models)
    bic_al_suite.judge(models)


if __name__ == "__main__":
    main()
