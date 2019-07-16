import gym
import sciunit
import numpy as np
import capabilities

# TODO: convert single-subject interfaces to multi-subject ones

def simulate_single_env_single_model(env, model, n_trials, seed=0):
    """
    Simulate the evolution of an environment and a model for a fixed
    number of steps.

    Parameters
    ----------
    env : gym.Env
        Environment.

    model : sciunit.Model and capabilities.Interactive
        Single subject model.

    n_trials : int
        Number of trials to perform. In each trial, model predicts the
        last stimulus produced by the environment. Afterwards environment and
        then the model is updated.

    seed : int
        Random seed used to initialize the environment.

    Returns
    -------
    stimuli : list
        List of all the stimuli produced after each trial. This list does not
        contain the initial state of the environment. Hence, its size is n_trials.

    rewards : list
        List of rewards obtained after each trial.

    actions : list
        List of actions performed by the model in each trial.
    """
    assert isinstance(env, gym.Env)
    assert isinstance(model, sciunit.Model)
    assert isinstance(model, capabilities.Interactive)
    assert env.observation_space == model.observation_space, "Observation space must be the same between environment and the model"
    assert env.action_space == model.action_space, "Action space must be the same between environment and the model"
    
    initial_stimulus = env.reset()
    env.seed(seed)
    actions = []
    rewards = []
    stimuli = []

    stimuli.append(initial_stimulus)

    for i in range(n_trials):
        s = stimuli[-1]
        a = model.act(s)

        s_next, r, done, _ = env.step(a)

        model.update(s, r, a, done)
        
        actions.append(a)
        rewards.append(r)
        stimuli.append(s_next)
    
    env.close()

    return stimuli[1:], rewards, actions

def simulate_multi_env_single_model(env_iterable, model, n_trials, seed=0):
    """
    Simulate the evolution of a series of environments with a single model that
    is reset only at the beginning and then continuously updated.

    Parameters
    ----------
    env_iterable : iterable of gym.Env
        Sequence of environments to simulate in the same order as they are given.
        This iterable is used to construct a complete list of environments at the start.

    model : sciunit.Model and capabilities.Interactive
        Single subject model that is reset only at the beginning.

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
    stimuli : list
        List which is the result of concatenating resulting stimuli lists for each
        environment.

    rewards : list
        List which is the result of concatenating resulting rewards lists for each
        environment.

    actions : list
        List which is the result of concatenating resulting actions lists for each
        environment.

    See Also
    --------
    simulate_single_env_single_model
    """
    env_list = list(env_iterable)
    if isinstance(n_trials, int):
        n_trials_list = np.repeat(n_trials, len(env_list), dtype=int)
    else:
        n_trials_list = list(n_trials)
        if len(n_trials_list) != len(env_list):
            raise ValueError('n_trials must be int or iterable of same length as env_list')

    actions = []
    rewards = []
    stimuli = []
    model.reset()
    for n, env in zip(n_trials_list, env_list):
        s, r, a = simulate_single_env_single_model(env, model, n, seed)
        stimuli.extend(s)
        rewards.extend(r)
        actions.extend(a)
    
    return stimuli, rewards, actions

def simulate(env, model, n_trials, seed=0):
    """
    Simulate a single- or multi-subject on one or more environments and return the
    results as list of lists.

    Parameters
    ----------
    env : gym.Env or iterable of gym.Env
        A single environment or iterable of environments.
        Refer to simulate_multi_env_single_model

    model : sciunit.Model and capabilities.Interactive
        Single- or multi-subject model. If a multi-subject model is passed, then each
        individual model will be simulated separately. Refer to simulate_multi_env_single_model

    n_trials : int or iterable of int
        Refer to simulate_multi_env_single_model

    seed :
        Refer to simulate_multi_env_single_model

    Returns
    -------
    stimuli : list or list of list
        If a single-subject model is passed, returns a list of stimuli explained in
        simulate_multi_env_single_model. If a multi-subject model is passed, each element
        of this list is the resulting list from simulate_multi_env_single_model.
    
    rewards : list or list of list
        If a single-subject model is passed, returns a list of rewards explained in
        simulate_multi_env_single_model. If a multi-subject model is passed, each element
        of this list is the resulting list from simulate_multi_env_single_model.

    actions : list or list of list
        If a single-subject model is passed, returns a list of actions explained in
        simulate_multi_env_single_model. If a multi-subject model is passed, each element
        of this list is the resulting list from simulate_multi_env_single_model.

    See Also
    --------
    simulate_multi_env_single_model
    """
    try:
        it = iter(env)
    except TypeError:
        env = [env]

    actions = []
    rewards = []
    stimuli = []
    if hasattr(model, 'models'):
        for m in model.models:
            m.reset()
            s, r, a = simulate_multi_env_single_model(env, m, n_trials, seed)
            stimuli.append(s)
            rewards.append(r)
            actions.append(a)

    else:
        stimuli, rewards, actions = simulate_multi_env_single_model(env, model, n_trials, seed)

    return stimuli, rewards, actions

class MultiMetaInteractive(type):
    """
    MultiMetaInteractive is a metaclass for creating multi-subject models from
    interactive single-subject ones. The input single-subject model should
    implement all the requirements of an interactive model (see capabilities.Interactive).

    The classes created by this metaclass implement all four methods of an
    interactive method. In addition, each method takes an additional subject
    index as their first argument. This index is used to select the individual
    single-subject model to use. In this regard, the returned class is semantically
    similar to a list of single-subject models while also satisfying model class requirements.

    This metaclass is not intended to be used directly. Users should use 
    multi_from_single_interactive function for automatically generating multi-subject models
    from single-subject ones.

    See Also
    --------
    multi_from_single_interactive
    """
    def __new__(cls, name, bases, dct):
        single_cls = dct['single_cls']
        base_classes = (single_cls.__bases__)
        out_cls = super().__new__(cls, name, base_classes, dct)

        # TODO: is there a clean way to make this metaclass more generic?
        # Maybe we can define all the necessary multi_.+ methods automatically.

        def multi_init(self, param_list, *args, **kwargs):
            self.subject_models = []
            for param in param_list:
                self.subject_models.append(single_cls(*args, **kwargs, paras=param))
        out_cls.__init__ = multi_init

        def multi_predict(self, idx, *args, **kwargs):
            paras = self.subject_models[idx].paras
            return self.subject_models[idx].predict(paras=paras, *args, **kwargs)
        out_cls.predict = multi_predict

        def multi_update(self, idx, *args, **kwargs):
            paras = self.subject_models[idx].paras
            return self.subject_models[idx].update(paras=paras, *args, **kwargs)
        out_cls.update = multi_update

        def multi_act(self, idx, *args, **kwargs):
            paras = self.subject_models[idx].paras
            return self.subject_models[idx].act(paras=paras, *args, **kwargs)
        out_cls.act = multi_act

        def multi_reset(self, idx, *args, **kwargs):
            paras = self.subject_models[idx].paras
            return self.subject_models[idx].reset(paras=paras, *args, **kwargs)
        out_cls.reset = multi_reset

        return out_cls

def multi_from_single_interactive(single_cls):
    """
    Create an interactive multi-subject model from an interactive
    single-subject model.

    Parameters
    ----------
    single_cls : capabilities.Interactive
        A single-subject model class implementing capabilities.Interactive interface.

    Returns
    -------
    MultiSubjectModel
        A multi-subject model class implementing capabilities.Interactive interface.
        Each required method now takes a subject index as their first argument.
    """
    multi_cls_name = 'Multi' + single_cls.name
    return MultiMetaInteractive(multi_cls_name, (), {'single_cls': single_cls})
