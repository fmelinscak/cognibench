import numpy as np
import sciunit
import pandas as pd
from os import getcwd
from os.path import join as pathjoin

from ldmunit.models import associative_learning
from ldmunit.models.utils import multi_from_single_interactive
from ldmunit.testing import InteractiveAICTest, InteractiveBICTest, InteractiveTest
from ldmunit.utils import partialclass
import ldmunit.scores as scores
from read_example_data import get_simulation_data, get_model_params

DATA_PATH = "data"
# sciunit CWD directory should contain config.json file
sciunit.settings["CWD"] = getcwd()

NLLScore = partialclass(scores.NLLScore, min_score=0, max_score=1000)


def main():
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


def main_fake():
    rr_al = get_simulation_data(pathjoin(DATA_PATH, "multi-rr_al.csv"), 3, True)
    nll_rr_al_test = InteractiveTest(
        name="rr_al sim NLL", observation=rr_al, score_type=NLLScore
    )
    suite = sciunit.TestSuite([nll_rr_al_test], name="suite")
    MultiRwNormModel = multi_from_single_interactive(associative_learning.RwNormModel)
    param_list = get_model_params(pathjoin(DATA_PATH, "multi-rw_norm_prior.csv"))
    multi_rw_norm = MultiRwNormModel(param_list, n_obs=4, seed=42)
    multi_rw_norm.name = "rw_norm"
    suite.judge([multi_rw_norm])


if __name__ == "__main__":
    main()
