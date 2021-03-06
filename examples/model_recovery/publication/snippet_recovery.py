import numpy as np
import scipy.stats as stats
from cognibench.tasks import model_recovery
import cognibench.models.decision_making as decision_models
from cognibench.testing import InteractiveTest
from cognibench.envs import BanditEnv
from cognibench.scores import BICScore
from cognibench.utils import partialclass
import multiprocessing
import sciunit
from os import getcwd

sciunit.settings["CWD"] = getcwd()


def bic_kwargs_fn(model, obs, pred):
    return {"n_model_params": model.n_params(), "n_samples": len(obs["stimuli"])}


def sm_to_numpy(sm):
    n_rows, n_cols = sm.shape
    arr = np.empty(shape=(n_rows, n_cols), dtype=np.float64)
    for i in range(n_rows):
        for j in range(n_cols):
            arr[i, j] = sm.iat[i, j].score
    return arr


def do_one(seed):
    init_seed = seed + int(1e5)
    env_seed = seed + int(1e6)
    rng = np.random.RandomState(init_seed)
    n_action, n_obs = 2, 2

    rr = decision_models.RandomRespondModel(n_action=n_action, n_obs=n_obs, seed=seed)
    rr.init_paras()
    rr.set_paras_kw(bias=rng.uniform())

    nwsls = decision_models.NWSLSModel(n_action=n_action, n_obs=n_obs, seed=seed)
    nwsls.init_paras()
    nwsls.set_paras_kw(epsilon=rng.uniform())

    rw = decision_models.RWModel(n_action=n_action, n_obs=n_obs, seed=seed)
    rw.init_paras()
    rw.set_paras_kw(eta=rng.uniform(), beta=1 + rng.exponential(scale=1))

    ck = decision_models.CKModel(n_action=n_action, n_obs=n_obs, seed=seed)
    ck.init_paras()
    ck.set_paras_kw(eta_c=rng.uniform(), beta_c=1 + rng.exponential(scale=1))

    rwck = decision_models.RWCKModel(n_action=n_action, n_obs=n_obs, seed=seed)
    rwck.init_paras()
    rwck.set_paras_kw(
        eta=rng.uniform(),
        eta_c=rng.uniform(),
        beta=1 + rng.exponential(scale=1),
        beta_c=1 + rng.exponential(scale=1),
    )

    model_list = [rr, nwsls, rw, ck, rwck]

    env = BanditEnv(p_dist=[0.2, 0.8], seed=env_seed)
    test_class = partialclass(
        InteractiveTest,
        score_type=partialclass(BICScore, min_score=0, max_score=1000),
        fn_kwargs_for_score=bic_kwargs_fn,
    )

    _, score_matrix = model_recovery(model_list, env, test_class, n_trials=1000)
    sm_arr = sm_to_numpy(score_matrix)
    np.save(f"{seed}", sm_arr)
    min_indices = np.argmin(sm_arr, axis=0)
    return min_indices


seeds = range(1, 101)
with multiprocessing.Pool(34) as p:
    index_list = p.map(do_one, seeds)

cm = np.zeros(shape=(5, 5))
for min_indices in index_list:
    cm[min_indices, np.arange(5)] += 1
cm /= np.sum(cm, axis=0)
np.save("conf_mat.npy", cm)
print(cm)
