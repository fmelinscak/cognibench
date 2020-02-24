import numpy as np
from cognibench.tasks import model_recovery
import cognibench.models.decision_making as decision_models
from cognibench.testing import InteractiveTest
from cognibench.envs import BanditEnv
from cognibench.scores import BICScore
from cognibench.utils import partialclass
import multiprocessing


def bic_kwargs_fn(model, obs, pred):
    return {"n_model_params": model.n_params(), "n_samples": len(obs)}


n_action, n_obs = 2, 2
model_list = [
    decision_models.RWModel(n_action=n_action, n_obs=n_obs),
    decision_models.CKModel(n_action=n_action, n_obs=n_obs),
    decision_models.RWCKModel(n_action=n_action, n_obs=n_obs),
    decision_models.NWSLSModel(n_action=n_action, n_obs=n_obs),
]
for model in model_list:
    model.init_paras()

env = BanditEnv(p_dist=[0.2, 0.8])
test_class = partialclass(
    InteractiveTest,
    score_type=partialclass(BICScore, min_score=0, max_score=1000),
    fn_kwargs_for_score=bic_kwargs_fn,
)

N = len(model_list)


def sm_to_numpy(sm):
    n_rows, n_cols = sm.shape
    arr = np.empty(shape=(n_rows, n_cols), dtype=np.float64)
    for i in range(n_rows):
        for j in range(n_cols):
            arr[i, j] = sm.iat[i, j].score
    return arr


def do_one(seed):
    _, score_matrix = model_recovery(
        model_list, env, test_class, n_trials=250, seed=seed
    )
    sm_arr = sm_to_numpy(score_matrix)
    np.save(f"{seed}", sm_arr)
    min_indices = np.argmin(sm_arr, axis=0)
    return min_indices


seeds = range(1, 51)
with multiprocessing.Pool(4) as p:
    index_list = p.map(do_one, seeds)

cm = np.zeros(shape=(N, N))
for min_indices in index_list:
    cm[min_indices, np.arange(N)] += 1
cm /= np.sum(cm, axis=0)
np.save("conf_mat.npy", cm)
print(cm)
