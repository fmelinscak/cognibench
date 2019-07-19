import gym
import sciunit
import numpy as np
from ldmunit.capabilities import Interactive


def simulate_single_env_single_model(env, multimodel, subject_idx, n_trials, seed=0):
    """
    Simulate the evolution of an environment and a multi-subject model for a
    fixed number of steps. Each subject model in the given multi-subject model
    is simulated independently using the same initial environment and number
    of trials.

    Parameters
    ----------
    env : :class:`gym.Env`
        Environment.

    multimodel : :class:`sciunit.models.Model` and :class:`ldmunit.capabilities.Interactive`
        Multi-subject model. If you have a single-subject model, refer to
        multi_from_single_interactive .

    subject_idx : int
        Subject index of the model to simulate.

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
    subject_actions = []
    subject_rewards = []
    subject_stimuli = []
    initial_stimulus = env.reset()
    env.seed(seed)
    subject_stimuli.append(initial_stimulus)
    for i in range(n_trials):
        s = subject_stimuli[-1]
        a = multimodel.act(subject_idx, s)
        s_next, r, done, _ = env.step(a)
        multimodel.update(subject_idx, s, r, a, done)
        subject_actions.append(a)
        subject_rewards.append(r)
        subject_stimuli.append(s_next)
    env.close()

    return subject_stimuli[1:], subject_rewards, subject_actions


def simulate_multi_env_multi_model(env_iterable, multimodel, n_trials, seed=0):
    """
    Simulate the evolution of a multiple of environments with multi-subject model.
    Each subject specific model is reset only at the beginning and then continuously
    updated using the given sequence of environments.

    Parameters
    ----------
    env_iterable : iterable of :class:`gym.Env`
        Sequence of environments to simulate in the same order as they are given.
        This iterable is used to construct a complete list of environments at the start.

    multimodel : :class:`sciunit.models.Model` and :class:`ldmunit.capabilities.Interactive`
        Multi-subject model. Each subject-specific model is reset only once before simulating
        all the environments one after another. If you have a single-subject model, refer to
        :func:`multi_from_single_interactive` .

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
        Each element of this list is a subject list which is the result of concatenating
        resulting stimuli lists for each environment.

    rewards : list of list
        Each element of this list is a subject list which is the result of concatenating
        resulting rewards lists for each environment.

    actions : list of list
        Each element of this list is a subject list which is the result of concatenating
        resulting actions lists for each environment.

    See Also
    --------
    simulate_single_env_single_model
    """
    env_list = list(env_iterable)
    if np.issubdtype(type(n_trials), np.integer):
        n_trials_list = np.repeat(n_trials, len(env_list))
    else:
        n_trials_list = list(n_trials)
        assert all(np.issubdtype(type(x), np.integer) for x in n_trials_list), 'All elements of n_trials must be int'
        assert len(n_trials_list) == len(env_list), 'n_trials must be int or iterable of same length as env_list'

    stimuli = []
    rewards = []
    actions = []
    n_models = len(multimodel.subject_models)
    for subject_idx in range(n_models):
        multimodel.reset(subject_idx)
        subject_stimuli = []
        subject_rewards = []
        subject_actions = []
        for n, env in zip(n_trials_list, env_list):
            s, r, a = simulate_single_env_single_model(env, multimodel, subject_idx, n, seed)
            subject_stimuli.extend(s)
            subject_rewards.extend(r)
            subject_actions.extend(a)
        stimuli.append(subject_stimuli)
        rewards.append(subject_rewards)
        actions.append(subject_actions)

    return stimuli, rewards, actions


class MultiMetaInteractive(type):
    """
    MultiMetaInteractive is a metaclass for creating multi-subject models from
    interactive single-subject ones. The input single-subject model should
    implement all the requirements of an interactive model (see :class:`ldmunit.capabilities.Interactive`).

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
        single_cls = bases[0]
        base_classes = (single_cls.__bases__)
        out_cls = super().__new__(cls, name, base_classes, dct)

        # TODO: is there a clean way to make this metaclass more generic?
        # Maybe we can define all the necessary multi_.+ methods automatically.

        def multi_init(self, param_list, *args, **kwargs):
            self.subject_models = []
            for param_dict in param_list:
                self.subject_models.append(single_cls(*args, **param_dict, **kwargs))

        out_cls.__init__ = multi_init

        def multi_predict(self, idx, *args, **kwargs):
            return self.subject_models[idx].predict(*args, **kwargs)

        out_cls.predict = multi_predict

        def multi_update(self, idx, *args, **kwargs):
            return self.subject_models[idx].update(*args, **kwargs)

        out_cls.update = multi_update

        def multi_act(self, idx, *args, **kwargs):
            return self.subject_models[idx].act(*args, **kwargs)

        out_cls.act = multi_act

        def multi_reset(self, idx, *args, **kwargs):
            return self.subject_models[idx].reset(*args, **kwargs)

        out_cls.reset = multi_reset

        return out_cls


def multi_from_single_interactive(single_cls):
    """
    Create an interactive multi-subject model from an interactive
    single-subject model.

    Parameters
    ----------
    single_cls : :class:`ldmunit.capabilities.Interactive`
        A single-subject model class implementing capabilities.Interactive interface.

    Returns
    -------
    MultiSubjectModel
        A multi-subject model class implementing :class:`ldmunit.capabilities.Interactive` interface.
        Each required method now takes a subject index as their first argument.
    """
    multi_cls_name = 'Multi' + single_cls.__name__
    return MultiMetaInteractive(multi_cls_name, (single_cls, ), {
        'name': single_cls.name,
        '__doc__': single_cls.__doc__
    })
