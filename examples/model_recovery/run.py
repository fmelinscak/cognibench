import numpy as np

from os import getcwd
import sciunit
from ldmunit.tasks import model_recovery
from ldmunit.models.associative_learning import (
    RwNormModel,
    KrwNormModel,
    BetaBinomialModel,
    LSSPDModel,
)
from ldmunit.models.utils import multi_from_single_cls
from ldmunit.testing import InteractiveTest
from ldmunit.envs import ClassicalConditioningEnv
from ldmunit.scores import NLLScore
from ldmunit.utils import partialclass

sciunit.settings["CWD"] = getcwd()
N_OBS = 5
models_params = [
    (RwNormModel, dict(w=0.6964, eta=0.2561, sigma=0.2268, b0=0.5513, b1=10.7194)),
    (
        KrwNormModel,
        dict(
            w=0.6964,
            sigma=0.2861,
            b0=0.2268,
            b1=0.5513,
            sigmaWInit=2.0533,
            tauSq=1.5266,
            sigmaRSq=2.6664,
            alpha=0.6848,
        ),
    ),
    (
        BetaBinomialModel,
        dict(intercept=0.6964, slope=0.2861, mix_coef=0.2268, sigma=0.5513),
    ),
    (
        LSSPDModel,
        dict(
            w=0.6964,
            alpha=0.2861,
            b0=0.2268,
            b1=0.5513,
            mix_coef=0.7194,
            eta=0.2531,
            kappa=0.9807,
            sigma=0.6848,
        ),
    ),
]

env_stimuli = [
    (np.array([0, 0, 0, 0, 0]), 0.1, 0.3),
    (np.array([0, 1, 0, 0, 1]), 0.4, 0.5),
    (np.array([0, 1, 1, 1, 0]), 0.25, 0.8),
    (np.array([1, 1, 1, 1, 1]), 0.25, 0.45),
]


def main_recovery_single():
    model_list = [ctor(n_obs=N_OBS, **dictionary) for ctor, dictionary in models_params]

    stimuli, p_stimuli, p_reward = zip(*env_stimuli)
    env = ClassicalConditioningEnv(
        stimuli=stimuli, p_stimuli=p_stimuli, p_reward=p_reward
    )
    test_cls = partialclass(
        InteractiveTest,
        multi_subject=False,
        score_type=partialclass(NLLScore, min_score=0, max_score=1e4),
    )
    suite, sm = model_recovery(model_list, env, test_cls, n_trials=25, seed=42)
    print("Single subject model recovery result matrix")
    print(sm.to_string())


def main_recovery_multi():
    n_subj = 5
    model_list = []
    for model, param in models_params:
        param_list = n_subj * [param]
        multimodel = multi_from_single_cls(model)
        model_list.append(multimodel(param_list, n_obs=N_OBS))

    stimuli, p_stimuli, p_reward = zip(*env_stimuli)
    env = ClassicalConditioningEnv(
        stimuli=stimuli, p_stimuli=p_stimuli, p_reward=p_reward
    )
    test_cls = partialclass(
        InteractiveTest,
        multi_subject=True,
        score_type=partialclass(NLLScore, min_score=0, max_score=1e4),
        score_aggr_fn=np.mean,
    )
    suite, sm = model_recovery(model_list, env, test_cls, n_trials=25, seed=42)
    print("Multi subject model recovery result matrix")
    print(sm.to_string())


if __name__ == "__main__":
    main_recovery_single()
    main_recovery_multi()
