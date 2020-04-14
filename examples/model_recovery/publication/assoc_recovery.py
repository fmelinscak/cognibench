import numpy as np
import scipy.stats as stats
from cognibench.tasks import model_recovery
import cognibench.models.associative_learning as assoc_models
from cognibench.testing import InteractiveTest
from cognibench.envs import ClassicalConditioningEnv
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


DISTINCT_STIMULI = np.array([[0.0, 1.0], [1.0, 0.0]])


def do_one(seed):
    init_seed = seed + int(1e5)
    env_seed = seed + int(1e6)
    rng = np.random.RandomState(init_seed)
    n_action, n_obs = 2, 2

    rw = assoc_models.RwNormModel(n_action=n_action, n_obs=n_obs, seed=init_seed)
    rw.init_paras()
    rw.set_seed(seed)

    krw = assoc_models.KrwNormModel(n_action=n_action, n_obs=n_obs, seed=init_seed)
    krw.init_paras()
    krw.set_seed(seed)

    lsspd = assoc_models.LSSPDModel(n_action=n_action, n_obs=n_obs, seed=init_seed)
    lsspd.init_paras()
    lsspd.set_seed(seed)

    bb = assoc_models.BetaBinomialModel(
        n_action=n_action,
        n_obs=n_obs,
        distinct_stimuli=DISTINCT_STIMULI,
        seed=init_seed,
    )
    bb.init_paras()
    bb.set_seed(seed)

    model_list = [rw, krw, lsspd, bb]

    env = ClassicalConditioningEnv(
        stimuli=DISTINCT_STIMULI, p_stimuli=[0.6, 0.4], p_reward=[0.3333, 0.0]
    )
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

cm = np.zeros(shape=(4, 4))
for min_indices in index_list:
    cm[min_indices, np.arange(4)] += 1
cm /= np.sum(cm, axis=0)
np.save("conf_mat.npy", cm)
print(cm)
