from ldmunit import simulation
from ldmunit.utils import is_arraylike
from ldmunit.capabilities import MultiSubjectModel
from ldmunit.logging import logger
import sciunit
import copy


def model_recovery(
    model_list, env, interactive_test_cls, n_trials=50, seed=None, **kwargs
):
    """
    For each of the given models,
      simulate the model with the environment
      test all the models against the simulated data

    For N models, this function produces N*N score values.

    Parameters
    ----------

    Returns
    -------
    """
    match, multi = _check_cardinalities_and_return(model_list)
    if not match:
        raise ValueError(
            "Models in model_recovery must be all single subject or all multiple subject"
        )
    if multi and not is_arraylike(env):
        env = [copy.deepcopy(env) for _ in range(multi.n_subjects)]

    sim_fun = simulation.simulate_multienv_multimodel if multi else simulation.simulate
    test_list = []
    for model in model_list:
        logger().info(f"model_recovery : Simulating model {model} against env {env}")
        stimuli, rewards, actions = sim_fun(env, model, n_trials, seed=seed)
        if multi:
            obs = []
            for subj_stimuli, subj_rewards, subj_actions in zip(
                stimuli, rewards, actions
            ):
                obs.append(
                    {
                        "stimuli": subj_stimuli,
                        "rewards": subj_rewards,
                        "actions": subj_actions,
                    }
                )
        else:
            obs = {"stimuli": stimuli, "rewards": rewards, "actions": actions}

        test_list.append(
            interactive_test_cls(observation=obs, name=f"Ground truth: {model.name}")
        )

    suite = sciunit.TestSuite(test_list, name="Model recovery test suite")
    score_matrix = suite.judge(model_list)
    return suite, score_matrix


def param_recovery():
    pass


def _check_cardinalities_and_return(model_list):
    n_single = 0
    n_multi = 0
    match = True
    multi = False
    for model in model_list:
        if isinstance(model, MultiSubjectModel):
            n_multi += 1
            multi = True
        else:
            n_single += 1
        if n_single > 0 and n_multi > 0:
            match = False
            break
    return match, multi
