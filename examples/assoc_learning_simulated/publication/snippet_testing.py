import sciunit
import pandas as pd
import numpy as np
import os
import cognibench.models.associative_learning as assoc_models
from cognibench.testing import InteractiveTest
from cognibench.utils import partialclass
from cognibench.models.utils import multi_from_single_cls as multi_subj
import cognibench.scores as scores
from read_example_data import get_simulation_data

sciunit.settings["CWD"] = os.getcwd()
# Constants
def aic_kwargs_fn(model, obs, pred):
    return {"n_model_params": model.n_params()}


SEED = 42
DATA_PATH = "../data"
N_SUBJECTS = 3
AICScore = partialclass(scores.AICScore, min_score=0, max_score=1000)
names_paths = [
    ("RwNorm Data", os.path.join(DATA_PATH, "multi-rw_norm.csv")),
    ("KrwNorm Data", os.path.join(DATA_PATH, "multi-krw_norm.csv")),
    ("LSSPD Data", os.path.join(DATA_PATH, "multi-lsspd.csv")),
    ("BB Data", os.path.join(DATA_PATH, "multi-beta_binomial.csv")),
]
# Define tests
test_list = []
distinct_stimuli = None
for test_name, path in names_paths:
    obs = get_simulation_data(path, N_SUBJECTS, True)
    distinct_curr = np.unique(obs[0]["stimuli"], axis=0)
    assert distinct_stimuli is None or (distinct_stimuli == distinct_curr).all()
    distinct_stimuli = distinct_curr
    test_list.append(
        InteractiveTest(
            name=f"{test_name}",
            observation=obs,
            score_type=AICScore,
            fn_kwargs_for_score=aic_kwargs_fn,
            multi_subject=True,
        )
    )
# Define models
MultiRwNormModel = multi_subj(assoc_models.RwNormModel)
MultiKrwNormModel = multi_subj(assoc_models.KrwNormModel)
MultiBetaBinomialModel = multi_subj(assoc_models.BetaBinomialModel)
MultiBetaBinomialModel.name = "BB"
MultiLSSPDModel = multi_subj(assoc_models.LSSPDModel)
model_list = [
    MultiRwNormModel(n_subj=N_SUBJECTS, n_obs=4, seed=SEED),
    MultiKrwNormModel(n_subj=N_SUBJECTS, n_obs=4, seed=SEED),
    MultiLSSPDModel(n_subj=N_SUBJECTS, n_obs=4, seed=SEED),
    MultiBetaBinomialModel(
        n_subj=N_SUBJECTS, n_obs=4, distinct_stimuli=distinct_stimuli, seed=SEED
    ),
]
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
