import sys
import numpy as np
import sciunit
import pandas as pd
from os import getcwd
from os.path import join as pathjoin

sys.path.append("..")
from ldmunit.models import decision_making
from ldmunit.models.utils import (
    multi_from_single_interactive_parametric as multi_subject,
)
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


def main():
    # Tests
    # -----------------------------------------------
    ck = get_simulation_data(pathjoin(DATA_PATH, "multi-ck.csv"), 3)
    nll_ck_test = InteractiveTest(
        multi_subject=True, name="CK sim NLL", observation=ck, score_type=NLLScore
    )
    aic_ck_test = InteractiveTest(
        multi_subject=True,
        name="CK sim AIC",
        observation=ck,
        score_type=AICScore,
        fn_kwargs_for_score=aic_kwargs_fn,
    )
    bic_ck_test = InteractiveTest(
        multi_subject=True,
        name="CK sim BIC",
        observation=ck,
        score_type=BICScore,
        fn_kwargs_for_score=bic_kwargs_fn,
    )

    rw = get_simulation_data(pathjoin(DATA_PATH, "multi-rw.csv"), 3)
    nll_rw_test = InteractiveTest(
        multi_subject=True, name="RW sim NLL", observation=rw, score_type=NLLScore
    )
    aic_rw_test = InteractiveTest(
        multi_subject=True,
        name="RW sim AIC",
        observation=rw,
        score_type=AICScore,
        fn_kwargs_for_score=aic_kwargs_fn,
    )
    bic_rw_test = InteractiveTest(
        multi_subject=True,
        name="RW sim BIC",
        observation=rw,
        score_type=BICScore,
        fn_kwargs_for_score=bic_kwargs_fn,
    )

    rwck = get_simulation_data(pathjoin(DATA_PATH, "multi-rwck.csv"), 3)
    nll_rwck_test = InteractiveTest(
        multi_subject=True, name="RWCK sim NLL", observation=rwck, score_type=NLLScore
    )
    aic_rwck_test = InteractiveTest(
        multi_subject=True,
        name="RWCK sim AIC",
        observation=rwck,
        score_type=AICScore,
        fn_kwargs_for_score=aic_kwargs_fn,
    )
    bic_rwck_test = InteractiveTest(
        multi_subject=True,
        name="RWCK sim BIC",
        observation=rwck,
        score_type=BICScore,
        fn_kwargs_for_score=bic_kwargs_fn,
    )

    rr = get_simulation_data(pathjoin(DATA_PATH, "multi-random-responding.csv"), 3)
    nll_rr_test = InteractiveTest(
        multi_subject=True, name="RR sim NLL", observation=rr, score_type=NLLScore
    )
    aic_rr_test = InteractiveTest(
        multi_subject=True,
        name="RR sim AIC",
        observation=rr,
        score_type=AICScore,
        fn_kwargs_for_score=aic_kwargs_fn,
    )
    bic_rr_test = InteractiveTest(
        multi_subject=True,
        name="RR sim BIC",
        observation=rr,
        score_type=BICScore,
        fn_kwargs_for_score=bic_kwargs_fn,
    )

    nwsls = get_simulation_data(pathjoin(DATA_PATH, "multi-nwsls.csv"), 3)
    nll_nwsls_test = InteractiveTest(
        multi_subject=True, name="NWSLS sim NLL", observation=nwsls, score_type=NLLScore
    )
    aic_nwsls_test = InteractiveTest(
        multi_subject=True,
        name="NWSLS sim AIC",
        observation=nwsls,
        score_type=AICScore,
        fn_kwargs_for_score=aic_kwargs_fn,
    )
    bic_nwsls_test = InteractiveTest(
        multi_subject=True,
        name="NWSLS sim BIC",
        observation=nwsls,
        score_type=BICScore,
        fn_kwargs_for_score=bic_kwargs_fn,
    )

    nll_suite = sciunit.TestSuite(
        [nll_ck_test, nll_rw_test, nll_rwck_test, nll_nwsls_test], name="NLL suite"
    )
    aic_suite = sciunit.TestSuite(
        [aic_ck_test, aic_rw_test, aic_rwck_test, aic_nwsls_test], name="AIC suite"
    )
    bic_suite = sciunit.TestSuite(
        [bic_ck_test, bic_rw_test, bic_rwck_test, bic_nwsls_test], name="BIC suite"
    )

    # Decision making models
    # -----------------------------------------------
    MultiRWModel = multi_subject(decision_making.RWModel)
    MultiCKModel = multi_subject(decision_making.CKModel)
    MultiRWCKModel = multi_subject(decision_making.RWCKModel)
    MultiNWSLSModel = multi_subject(decision_making.NWSLSModel)
    n_action, n_obs = 3, 3

    param_list = get_model_params(pathjoin(DATA_PATH, "multi-ck_prior.csv"))
    multi_ck = MultiCKModel(param_list, n_action=n_action, n_obs=n_obs)
    multi_ck.name = "ck"

    param_list = get_model_params(pathjoin(DATA_PATH, "multi-rw_prior.csv"))
    multi_rw = MultiRWModel(param_list, n_action=n_action, n_obs=n_obs)
    multi_rw.name = "rw"

    param_list = get_model_params(pathjoin(DATA_PATH, "multi-rwck_prior.csv"))
    multi_rwck = MultiRWCKModel(param_list, n_action=n_action, n_obs=n_obs)
    multi_rwck.name = "rwck"

    param_list = get_model_params(pathjoin(DATA_PATH, "multi-nwsls_prior.csv"))
    multi_nwsls = MultiNWSLSModel(param_list, n_action=n_action, n_obs=n_obs)
    multi_nwsls.name = "nwsls"

    # Run test suites
    # -----------------------------------------------
    nll_suite.judge([multi_ck, multi_rw, multi_rwck, multi_nwsls])
    aic_suite.judge([multi_ck, multi_rw, multi_rwck, multi_nwsls])
    bic_suite.judge([multi_ck, multi_rw, multi_rwck, multi_nwsls])


if __name__ == "__main__":
    main()
