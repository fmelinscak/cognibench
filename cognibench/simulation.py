import numpy as np
from functools import partial
from itertools import starmap
from cognibench.models import CNBModel, CNBAgent
from cognibench.envs import CNBEnv
from cognibench.models.utils import single_from_multi_obj, reverse_single_from_multi_obj
from cognibench.capabilities import ActionSpace, ObservationSpace, Interactive
from cognibench.logging import logger
from cognibench import settings


def simulate(env, model_or_agent, n_trials, check_env_model=True):
    """
    Simulate the evolution of an environment and a model or an agent for a
    fixed number of steps.

    Parameters
    ----------
    env : :class:`gym.Env`
        Environment.

    model : :class:`cognibench.models.CNBModel` and :class:`cognibench.capabilities.Interactive` or :class:`cognibench.models.CNBAgent`
        Agent or an already fitted model.

    n_trials : int
        Number of trials to perform. In each trial, model acts on the
        last stimulus produced by the environment. Afterwards environment and
        then the model is updated.

    check_env_model : bool
        Whether to check if the model/agent and the environment has matching action and observation spaces.

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
    if check_env_model and not _model_env_capabilities_match(env, model_or_agent):
        error_msg = f"simulate : Env {env} and model {model_or_agent} action and observation spaces aren't the same!"
        logger().error(error_msg)
        if settings["CRASH_EARLY"]:
            raise ValueError(error_msg)
        return [], [], []

    actions = []
    rewards = []
    stimuli = []
    initial_stimulus = env.reset()
    stimuli.append(initial_stimulus)
    for i in range(n_trials):
        s = stimuli[-1]
        a = model_or_agent.act(s)
        s_next, r, done, _ = env.step(a)
        model_or_agent.update(s, r, a, done)
        env.update(s, r, a, done)
        actions.append(a)
        rewards.append(r)
        stimuli.append(s_next)
    env.close()

    return stimuli[1:], rewards, actions


def simulate_multienv_multimodel(
    env_iterable, multimodel, n_trials, check_env_model=True
):
    """
    Simulate the evolution of multiple environments with multi-subject model.
    Each subject model gets their own environment.

    Parameters
    ----------
    env_iterable : iterable of :class:`gym.Env`
        Sequence of environments to simulate in the same order as they are given.
        This iterable is used to construct a complete list of environments at the start.

    multimodel : :class:`sciunit.models.Model`, :class:`cognibench.capabilities.Interactive`, :class:`cognibench.capabilities.MultiSubjectModel`
        Multi-subject model.

    n_trials : int or iterable of int
        Number of trials to perform. If int, then each environment is simulated
        for n_trials many steps before proceeding to the next environment. If
        iterable, n_trials must contain the number of steps for each environment
        in the same order. In this case, length of n_trials must be the same as
        that of env_iterable.

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
    :py:func:`simulate`
    """
    env_list = list(env_iterable)
    if check_env_model:
        for i, env in enumerate(env_list):
            model_i = single_from_multi_obj(multimodel, i)
            do_match = _model_env_capabilities_match(env, model_i)
            multimodel = reverse_single_from_multi_obj(model_i)
            if not do_match:
                error_msg = f"simulate : Env {env} and model {multimodel} action and observation spaces aren't the same!"
                logger().error(error_msg)
                if settings["CRASH_EARLY"]:
                    raise ValueError(error_msg)
                return [], [], []

    if np.issubdtype(type(n_trials), np.integer):
        n_trials_list = np.repeat(n_trials, len(env_list))
    else:
        n_trials_list = list(n_trials)
        assert len(n_trials_list) == len(
            env_list
        ), "n_trials must be int or iterable of same length as env_list"

    def sim_i(multimodel, idx):
        model_i = single_from_multi_obj(multimodel, idx)
        out_tuple = simulate(
            env_list[idx], model_i, n_trials_list[idx], check_env_model=False
        )
        multimodel = reverse_single_from_multi_obj(model_i)
        return out_tuple

    all_out = map(partial(sim_i, multimodel), range(len(env_list)))
    stimuli, rewards, actions = zip(*all_out)

    return stimuli, rewards, actions


def _model_env_capabilities_match(env, model_or_agent):
    """
    Check if capabilities, action spaces and observation spaces of the environment and model/agent matches.

    Parameters
    ----------
    env : `cognibench.env.CNBEnv`
        Environment.

    model_or_agent : `cognibench.models.CNBModel` or `cognibench.models.CNBAgent`
        Model or agent.

    Returns
    -------
    is_match : bool
        `True` if the environment and the model/agent matches; `False` otherwise.
    """
    # check types
    is_model = isinstance(model_or_agent, CNBModel)
    type_checks = [
        (model_or_agent, Interactive) if is_model else (model_or_agent, CNBAgent),
        (env, CNBEnv),
        (model_or_agent, ActionSpace),
        (model_or_agent, ObservationSpace),
        (env, ActionSpace),
        (env, ObservationSpace),
    ]
    for obj, cls in type_checks:
        if not isinstance(obj, cls):
            error_msg = f"_model_env_capabilities_match : Model {obj} is not an instance of {cls}!"
            logger().error(error_msg)
            if settings["CRASH_EARLY"]:
                raise ValueError(error_msg)
            return False

    env_action_space = type(env.get_action_space())
    env_obs_space = type(env.get_observation_space())
    model_action_space = type(model_or_agent.get_action_space())
    model_obs_space = type(model_or_agent.get_observation_space())
    return env_action_space == model_action_space and env_obs_space == model_obs_space
