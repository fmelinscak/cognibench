import numpy as np
from functools import partial
from ldmunit.models.utils import single_from_multi_obj, reverse_single_from_multi_obj


def simulate(env, model, n_trials, seed=None):
    """
    Simulate the evolution of an environment and a model for a
    fixed number of steps. Each subject model in the given multi-subject model
    is simulated independently using the same initial environment and number
    of trials.

    Parameters
    ----------
    env : :class:`gym.Env`
        Environment.

    model : :class:`ldmunit.models.LDMModel` and :class:`ldmunit.capabilities.Interactive`
        Model.

    n_trials : int
        Number of trials to perform. In each trial, model acts on the
        last stimulus produced by the environment. Afterwards environment and
        then the model is updated.

    seed : int
        Random seed used to initialize the environment.

    Returns
    -------
    stimuli : list
        List of all the stimuli produced after each trial. This list does not
        contain the initial state of the environment. Hence, its size is `n_trials`.

    rewards : list
        List of rewards obtained after each trial.

    actions : list
        List of actions performed by the model in each trial.
    """
    actions = []
    rewards = []
    stimuli = []
    env.seed(seed)
    initial_stimulus = env.reset()
    stimuli.append(initial_stimulus)
    for i in range(n_trials):
        s = stimuli[-1]
        a = model.act(s)
        s_next, r, done, _ = env.step(a)
        model.update(s, r, a, done)
        env.update(s, r, a, done)
        actions.append(a)
        rewards.append(r)
        stimuli.append(s_next)
    env.close()

    return stimuli[1:], rewards, actions


def simulate_multienv_multimodel(env_iterable, multimodel, n_trials, seed=None):
    """
    Simulate the evolution of multiple environments with multi-subject model.
    Each subject model gets their own environment.

    Parameters
    ----------
    env_iterable : iterable of :class:`gym.Env`
        Sequence of environments to simulate in the same order as they are given.
        This iterable is used to construct a complete list of environments at the start.

    multimodel : :class:`sciunit.models.Model` and :class:`ldmunit.capabilities.Interactive`
        Multi-subject model.

    n_trials : int or iterable of int
        Number of trials to perform. If int, then each environment is simulated
        for n_trials many steps before proceeding to the next environment. If
        iterable, n_trials must contain the number of steps for each environment
        in the same order. In this case, length of n_trials must be the same as
        that of env_iterable.

    seed : int
        Random seed used to initialize every environment.

    Returns
    -------
    stimuli : list of list
        Each element of this list is a subject list which is the result of the simulation with the
        corresponding environment.

    rewards : list of list
        Each element of this list is a reward list which is the result of the simulation with the
        corresponding environment.

    actions : list of list
        Each element of this list is a action list which is the result of the simulation with the
        corresponding environment.

    See Also
    --------
    simulate_single_env_single_model
    """
    env_list = list(env_iterable)
    if np.issubdtype(type(n_trials), np.integer):
        n_trials_list = np.repeat(n_trials, len(env_list))
    else:
        n_trials_list = list(n_trials)
        assert len(n_trials_list) == len(
            env_list
        ), "n_trials must be int or iterable of same length as env_list"

    def sim_i(multimodel, idx):
        model_i = single_from_multi_obj(multimodel, idx)
        out_tuple = simulate(env_list[idx], model_i, n_trials_list[idx], seed)
        multimodel = reverse_single_from_multi_obj(model_i)
        return out_tuple

    all_out = map(partial(sim_i, multimodel), range(len(env_list)))
    stimuli, rewards, actions = zip(*all_out)

    return stimuli, rewards, actions
