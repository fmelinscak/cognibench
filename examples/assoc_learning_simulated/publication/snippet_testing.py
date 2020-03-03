import sciunit
import pandas as pd
import numpy as np
import os
import cognibench.models.associative_learning as assoc_models
from cognibench.testing import InteractiveTest
from cognibench.simulation import simulate
from cognibench.envs import ClassicalConditioningEnv
from cognibench.utils import partialclass
from cognibench.models.utils import multi_from_single_cls as multi_subj
import cognibench.scores as scores
from read_example_data import get_simulation_data

# Constants
sciunit.settings["CWD"] = os.getcwd()
SEED = 42
AICScore = partialclass(scores.AICScore, min_score=0, max_score=1000)
DISTINCT_STIMULI = np.array([[0, 1], [1, 0]], dtype=np.float64)


def aic_kwargs_fn(model, obs, pred):
    return {"n_model_params": model.n_params()}


def exp_twostage_twocueonly(n_trials_all, probs_all, model):
    obs = {"stimuli": [], "actions": [], "rewards": []}
    for n_trials, probs in zip(n_trials_all, probs_all):
        p_a, p_us_a, p_us_b = probs
        env = ClassicalConditioningEnv(
            stimuli=DISTINCT_STIMULI,
            p_stimuli=[p_a, 1 - p_a],
            p_reward=[p_us_a, p_us_b],
        )
        stimuli, rewards, actions = simulate(env, model, n_trials)
        obs["stimuli"] += stimuli
        obs["rewards"] += rewards
        obs["actions"] += actions
    return obs


def get_sim_data(model):
    n_trials_all = [60, 60]
    probs_all = [(0.6, 0.3333, 0), (0.4103, 0, 0.3043)]
    return exp_twostage_twocueonly(n_trials_all, probs_all, model)


model_list = [
    assoc_models.RwNormModel(n_obs=2, seed=SEED),
    assoc_models.KrwNormModel(n_obs=2, seed=SEED),
    assoc_models.LSSPDModel(n_obs=2, seed=SEED),
    assoc_models.BetaBinomialModel(
        n_obs=2, distinct_stimuli=DISTINCT_STIMULI, seed=SEED
    ),
]
names_data = [(f"{model.name} Data", get_sim_data(model)) for model in model_list]
# Define tests
test_list = []
for test_name, obs in names_data:
    test_list.append(
        InteractiveTest(
            name=f"{test_name}",
            observation=obs,
            score_type=AICScore,
            fn_kwargs_for_score=aic_kwargs_fn,
            multi_subject=False,
        )
    )
# Define models
# Define suite and judge
suite = sciunit.TestSuite(test_list, name="Associative learning suite")
score_matrix = suite.judge(model_list)


def sm_to_numpy(sm):
    n_rows, n_cols = sm.shape
    arr = np.empty(shape=(n_rows, n_cols), dtype=np.float64)
    for i in range(n_rows):
        for j in range(n_cols):
            arr[i, j] = sm.iat[i, j].score
    return arr


sm_arr = sm_to_numpy(score_matrix)
df = pd.DataFrame(data=sm_arr, index=score_matrix.index, columns=score_matrix.columns)
df.to_csv("score_matrix.csv")
