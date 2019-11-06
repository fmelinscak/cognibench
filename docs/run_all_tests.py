import sys
import numpy as np
import sciunit
import pandas as pd
from os import getcwd
from os.path import join as pathjoin

sys.path.append("..")
from ldmunit.models import decision_making, associative_learning
from ldmunit.models.utils import multi_from_single_interactive
from ldmunit.testing import InteractiveAICTest, InteractiveBICTest, InteractiveTest
from ldmunit.utils import partialclass
import ldmunit.scores as scores
from read_example_data import get_simulation_data, get_model_params

DATA_PATH = pathjoin("..", "data")
# sciunit CWD directory should contain config.json file
sciunit.settings["CWD"] = getcwd()

NLLScore = partialclass(scores.NLLScore, min_score=0, max_score=1000)


def test_all_decision_making_models():
    # Tests
    # -----------------------------------------------
    ck = get_simulation_data(pathjoin(DATA_PATH, "multi-ck.csv"), 3)
    nll_ck_test = InteractiveTest(
        name="CK sim NLL", observation=ck, score_type=NLLScore
    )
    aic_ck_test = InteractiveAICTest(name="CK sim AIC", observation=ck)
    bic_ck_test = InteractiveBICTest(name="CK sim BIC", observation=ck)

    rw = get_simulation_data(pathjoin(DATA_PATH, "multi-rw.csv"), 3)
    nll_rw_test = InteractiveTest(
        name="RW sim NLL", observation=rw, score_type=NLLScore
    )
    aic_rw_test = InteractiveAICTest(name="RW sim AIC", observation=rw)
    bic_rw_test = InteractiveBICTest(name="RW sim BIC", observation=rw)

    rwck = get_simulation_data(pathjoin(DATA_PATH, "multi-rwck.csv"), 3)
    nll_rwck_test = InteractiveTest(
        name="RWCK sim NLL", observation=rwck, score_type=NLLScore
    )
    aic_rwck_test = InteractiveAICTest(name="RWCK sim AIC", observation=rwck)
    bic_rwck_test = InteractiveBICTest(name="RWCK sim BIC", observation=rwck)

    rr = get_simulation_data(pathjoin(DATA_PATH, "multi-random-responding.csv"), 3)
    nll_rr_test = InteractiveTest(
        name="RR sim NLL", observation=rr, score_type=NLLScore
    )
    aic_rr_test = InteractiveAICTest(name="RR sim AIC", observation=rr)
    bic_rr_test = InteractiveBICTest(name="RR sim BIC", observation=rr)

    nwsls = get_simulation_data(pathjoin(DATA_PATH, "multi-nwsls.csv"), 3)
    nll_nwsls_test = InteractiveTest(
        name="NWSLS sim NLL", observation=nwsls, score_type=NLLScore
    )
    aic_nwsls_test = InteractiveAICTest(name="NWSLS sim AIC", observation=nwsls)
    bic_nwsls_test = InteractiveBICTest(name="NWSLS sim BIC", observation=nwsls)

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
    MultiRWModel = multi_from_single_interactive(decision_making.RWModel)
    MultiCKModel = multi_from_single_interactive(decision_making.CKModel)
    MultiRWCKModel = multi_from_single_interactive(decision_making.RWCKModel)
    MultiNWSLSModel = multi_from_single_interactive(decision_making.NWSLSModel)
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


def test_all_associative_learning_models():
    # Tests
    # -----------------------------------------------
    rr_al = get_simulation_data(pathjoin(DATA_PATH, "multi-rr_al.csv"), 3, True)
    nll_rr_al_test = InteractiveTest(
        name="rr_al sim NLL", observation=rr_al, score_type=NLLScore
    )
    aic_rr_al_test = InteractiveAICTest(name="rr_al sim AIC", observation=rr_al)
    bic_rr_al_test = InteractiveBICTest(name="rr_al sim BIC", observation=rr_al)

    rw_norm = get_simulation_data(pathjoin(DATA_PATH, "multi-rw_norm.csv"), 3, True)
    nll_rw_norm_test = InteractiveTest(
        name="rw_norm sim NLL", observation=rw_norm, score_type=NLLScore
    )
    aic_rw_norm_test = InteractiveAICTest(name="rw_norm sim AIC", observation=rw_norm)
    bic_rw_norm_test = InteractiveBICTest(name="rw_norm sim BIC", observation=rw_norm)

    krw_norm = get_simulation_data(pathjoin(DATA_PATH, "multi-krw_norm.csv"), 3, True)
    nll_krw_norm_test = InteractiveTest(
        name="krw_norm sim NLL", observation=krw_norm, score_type=NLLScore
    )
    aic_krw_norm_test = InteractiveAICTest(
        name="krw_norm sim AIC", observation=krw_norm
    )
    bic_krw_norm_test = InteractiveBICTest(
        name="krw_norm sim BIC", observation=krw_norm
    )

    lsspd = get_simulation_data(pathjoin(DATA_PATH, "multi-lsspd.csv"), 3, True)
    nll_lsspd_test = InteractiveTest(
        name="lsspd sim NLL", observation=lsspd, score_type=NLLScore
    )
    aic_lsspd_test = InteractiveAICTest(name="lsspd sim AIC", observation=lsspd)
    bic_lsspd_test = InteractiveBICTest(name="lsspd sim BIC", observation=lsspd)

    bb = get_simulation_data(pathjoin(DATA_PATH, "multi-beta_binomial.csv"), 3, True)
    nll_bb_test = InteractiveTest(
        name="Beta Binomial sim NLL", observation=bb, score_type=NLLScore
    )
    aic_bb_test = InteractiveAICTest(name="Beta Binomial sim AIC", observation=bb)
    bic_bb_test = InteractiveBICTest(name="Beta Binomial sim BIC", observation=bb)

    nll_al_suite = sciunit.TestSuite(
        [
            nll_rr_al_test,
            nll_rw_norm_test,
            nll_krw_norm_test,
            nll_lsspd_test,
            nll_bb_test,
        ],
        name="NLL suite for associative learning",
    )
    aic_al_suite = sciunit.TestSuite(
        [
            aic_rr_al_test,
            aic_rw_norm_test,
            aic_krw_norm_test,
            aic_lsspd_test,
            aic_bb_test,
        ],
        name="AIC suite for associative learning",
    )
    bic_al_suite = sciunit.TestSuite(
        [
            bic_rr_al_test,
            bic_rw_norm_test,
            bic_krw_norm_test,
            bic_lsspd_test,
            bic_bb_test,
        ],
        name="BIC suite for associative learning",
    )

    # Associative learning models
    # -----------------------------------------------
    MultiRwNormModel = multi_from_single_interactive(associative_learning.RwNormModel)
    MultiKrwNormModel = multi_from_single_interactive(associative_learning.KrwNormModel)
    MultiBetaBinomialModel = multi_from_single_interactive(
        associative_learning.BetaBinomialModel
    )
    MultiLSSPDModel = multi_from_single_interactive(associative_learning.LSSPDModel)

    param_list = get_model_params(pathjoin(DATA_PATH, "multi-rw_norm_prior.csv"))
    multi_rw_norm = MultiRwNormModel(param_list, n_obs=4)
    multi_rw_norm.name = "rw_norm"

    param_list = get_model_params(pathjoin(DATA_PATH, "multi-krw_norm_prior.csv"))
    multi_krw_norm = MultiKrwNormModel(param_list, n_obs=4)
    multi_krw_norm.name = "krw_norm"

    param_list = get_model_params(pathjoin(DATA_PATH, "multi-lsspd_prior.csv"))
    multi_lsspd = MultiLSSPDModel(param_list, n_obs=4)
    multi_lsspd.name = "lsspd"

    param_list = get_model_params(pathjoin(DATA_PATH, "multi-beta_binomial_prior.csv"))
    multi_bb = MultiBetaBinomialModel(param_list, n_obs=4)
    multi_bb.name = "bb"

    # Run test suites
    # -----------------------------------------------
    nll_al_suite.judge([multi_rw_norm, multi_krw_norm, multi_lsspd, multi_bb])
    aic_al_suite.judge([multi_rw_norm, multi_krw_norm, multi_lsspd, multi_bb])
    bic_al_suite.judge([multi_rw_norm, multi_krw_norm, multi_lsspd, multi_bb])


if __name__ == "__main__":
    test_all_decision_making_models()
    test_all_associative_learning_models()
