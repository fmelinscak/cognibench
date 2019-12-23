import numpy as np

from ldmunit.simulation import simulate, simulate_multienv_multimodel
from ldmunit.models.decision_making import NWSLSModel, NWSLSAgent
from ldmunit.models.utils import multi_from_single_cls
from ldmunit.envs import BanditEnv


SEED = 42


def main_simulate_single():
    p_dist = [0, 0, 0.333, 0.333, 0.333, 0, 0]
    model = NWSLSModel(n_action=len(p_dist), n_obs=1, seed=SEED)
    env = BanditEnv(p_dist=p_dist)

    stimuli, rewards, actions = simulate(env, model, 100, seed=SEED)
    print("Single simulation results")
    print("Stimuli")
    print(stimuli)
    print("Rewards")
    print(rewards)
    print("Actions")
    print(actions)


def main_simulate_multi():
    p_dist = [
        [0, 0, 0.333, 0.333, 0.333, 0, 0],
        7 * [1 / 7],
        [0.025, 0.025, 0.3, 0.3, 0.3, 0.025, 0.025],
    ]
    epsilon_list = [{"epsilon": e} for e in [0, 1, 2]]

    MultiNWSLS = multi_from_single_cls(NWSLSModel)
    multimodel = MultiNWSLS(n_subj=3, n_action=7, n_obs=1, seed=SEED)
    envs = [BanditEnv(p_dist=p) for p in p_dist]

    stimuli, rewards, actions = simulate_multienv_multimodel(
        envs, multimodel, 100, seed=SEED
    )
    print("Multi simulation results")
    for i, (subj_stimuli, subj_rewards, subj_actions) in enumerate(
        zip(stimuli, rewards, actions)
    ):
        print(f"Subject {i}")
        print("Stimuli")
        print(subj_stimuli)
        print("Rewards")
        print(subj_rewards)
        print("Actions")
        print(subj_actions)


if __name__ == "__main__":
    main_simulate_single()
    main_simulate_multi()
