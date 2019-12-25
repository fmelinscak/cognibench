from ldmunit import simulation
from ldmunit.utils import is_arraylike
from ldmunit.capabilities import MultiSubjectModel
from ldmunit.logging import logger
import sciunit
import copy


def model_recovery(model_list, env, interactive_test_cls, n_trials=50, seed=None):
    """
    Perform model recovery task and return the results as a score matrix.

    Model recovery is performed as below:
        1. For each of the models in the given list
            a. Create simulated data from the model using the given environment
            b. Test all the models against this simulated data using the given test class
        2. Return the results as a score matrix

    Parameters
    ----------
    model_list : iterable
        List of models

    env : `ldmunit.env.LDMEnv`
        Environment to use while simulating the data

    interactive_test_cls : `ldmunit.testing.LDMTest`
        Test class to use when testing all the models against the simulated data created by one of the models.

    n_trials : int
        Number of simulation trials.

    seed : int
        Random seed to use.

    Returns
    -------
    suite : :class:`sciunit.TestSuite`
        Test suite class created during this function call. The suite object contains all the data (observations, predictions, etc.) generated
        during the procedure.

    score_matrix : :class:`sciunit.ScoreMatrix`
        Score matrix object containing the score of each model for each of the simulation rounds.
    """
    match, multi = _check_cardinalities_and_return(model_list)
    assert (
        match
    ), "Models in model_recovery must be all single subject or all multiple subject"
    if multi:
        subj_cnts = [m.n_subjects for m in model_list]
        assert (
            len(set(subj_cnts)) == 1
        ), "All models must have the same number of subjects"
        n_subjects = model_list[0].n_subjects

    env_name = env.name
    if multi and not is_arraylike(env):
        env = [copy.deepcopy(env) for _ in range(n_subjects)]

    sim_fun = simulation.simulate_multienv_multimodel if multi else simulation.simulate
    test_list = []
    for model in model_list:
        logger().info(
            f"model_recovery : Simulating model {model.name} against env {env_name}"
        )
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


def param_recovery(
    paras_list, model, env, n_runs=5, n_trials=50, seed=None,
):
    """
    Perform parameter recovery task and return all of the fitted parameter values.

    Parameter recovery is performed as below:
        1. For each of the parameter dictionaries in the given parameter list
            a. Create simulated data using the environment and the model object initialized to the current parameters
            b. Fit the model object `n_runs` many times to this simulated data and store each of the fitted parameters
        2. Return all of the parameter fits (a `list` of shape `(len(paras_list), n_runs)`)

    Parameters
    ----------
    paras_list : iterable
        List of parameter dictionaries. Each parameter dictionary should be compatible with the given model object.

    model : `ldmunit.models.LDMModel`
        Model object to use for parameter recovery task.

    env : `ldmunit.env.LDMEnv`
        Environment to use while simulating the data

    n_runs : int
        Number of fits to perform for each of the parameter dictionaries in `paras_list`.

    n_trials : int
        Number of simulation trials.

    seed : int
        Random seed to use.

    Returns
    -------
    results : list of list
        Each element of the list contains `n_runs` many dictionaries. Each dictionary is the result of the corresponding
        model fit.
    """
    out = []
    for i, paras in enumerate(paras_list):
        logger().info(f"param_recovery: Recovering parameters with index {i}")
        out_paras = []
        for _ in range(n_runs):
            model.set_paras(paras)
            stimuli, rewards, actions = simulation.simulate(
                env, model, n_trials, seed=seed
            )
            model.init_paras()
            model.reset()
            model.fit(stimuli, rewards, actions)
            out_paras.append(model.get_paras())
        out.append(out_paras)
    return out


def _check_cardinalities_and_return(model_list):
    """
    Parameters
    ----------
    model_list : list
        List of model objects.

    Returns
    -------
    match : bool
        `True` if the model cardinalities (single or multi) of every model in `model_list` matches; `False` otherwise.

    multi : bool
        `True` if the models in the given list are multi-subject models; `False` otherwise.
    """
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
