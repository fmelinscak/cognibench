import numpy as np
import scipy.stats as stats

from os import getcwd
import sciunit
from cognibench.tasks import param_recovery
from cognibench.models.associative_learning import LSSPDModel
from cognibench.testing import InteractiveTest
from cognibench.envs import ClassicalConditioningEnv
from cognibench.utils import partialclass

sciunit.settings["CWD"] = getcwd()
N_OBS = 5

env_stimuli = [
    (np.array([0, 0, 0, 0, 0]), 0.1, 0.3),
    (np.array([0, 1, 0, 0, 1]), 0.4, 0.5),
    (np.array([0, 1, 1, 1, 0]), 0.25, 0.8),
    (np.array([1, 1, 1, 1, 1]), 0.25, 0.45),
]


def main_recovery_single():
    model = LSSPDModel(n_obs=N_OBS)

    stimuli, p_stimuli, p_reward = zip(*env_stimuli)
    env = ClassicalConditioningEnv(
        stimuli=stimuli, p_stimuli=p_stimuli, p_reward=p_reward
    )
    paras_list = [
        {
            "w": stats.norm.rvs(size=N_OBS),
            "alpha": stats.norm.rvs(size=N_OBS),
            "sigma": stats.expon.rvs(),
            "b0": stats.norm.rvs(),
            "b1": stats.norm.rvs(size=N_OBS),
            "mix_coef": stats.uniform.rvs(),
            "eta": 1e-2,
            "kappa": 1e-2,
        },
        {
            "w": stats.norm.rvs(size=N_OBS),
            "alpha": stats.norm.rvs(size=N_OBS),
            "sigma": stats.expon.rvs(),
            "b0": stats.norm.rvs(),
            "b1": stats.norm.rvs(size=N_OBS),
            "mix_coef": stats.uniform.rvs(),
            "eta": 1e-3,
            "kappa": 1e-3,
        },
        {
            "w": stats.norm.rvs(size=N_OBS),
            "alpha": stats.norm.rvs(size=N_OBS),
            "sigma": stats.expon.rvs(),
            "b0": stats.norm.rvs(),
            "b1": stats.norm.rvs(size=N_OBS),
            "mix_coef": stats.uniform.rvs(),
            "eta": 1e-4,
            "kappa": 1e-4,
        },
    ]

    results = param_recovery(paras_list, model, env, n_runs=5, n_trials=50, seed=42)
    for paras, res in zip(paras_list, results):
        print("Recovery results for parameters:")
        print("--------------------------------")
        print(paras)
        for run_idx, run_res in enumerate(res):
            print(f"Run {run_idx}")
            print(run_res)


if __name__ == "__main__":
    main_recovery_single()
