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
# TODO: initialize the models by fitting them to some data, and then try to recover?
models = [
    RwNormModel,
    KrwNormModel,
    BetaBinomialModel,
    LSSPDModel,
]

env_stimuli = [
    (np.array([0, 0, 0, 0, 0]), 0.1, 0.3),
    (np.array([0, 1, 0, 0, 1]), 0.4, 0.5),
    (np.array([0, 1, 1, 1, 0]), 0.25, 0.8),
    (np.array([1, 1, 1, 1, 1]), 0.25, 0.45),
]


def main_recovery_single():
    model_list = []
    for ctor in models:
        model = ctor(n_obs=N_OBS)
        model.init_paras()
        model_list.append(model)

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
    for model in models:
        multimodel = multi_from_single_cls(model)
        model = multimodel(n_subj=n_subj, n_obs=N_OBS)
        for i in range(n_subj):
            model.init_paras(i)
        model_list.append(model)

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
