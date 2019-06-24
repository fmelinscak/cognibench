from scipy.optimize import minimize
import gym
import sciunit
import numpy as np

def loglike(model, stimuli, rewards, actions):
    
    res = 0
    model.reset()

    for s, r, a in zip(stimuli, rewards, actions):
        # compute choice probabilities
        log_prob = model.predict(s)
        # probability of the action
        res += log_prob(a)
        # update choice kernel and Q weights
        model.update(s, r, a, False)

    # fix for shape(1,)
    if isinstance(res, list) or isinstance(res, np.ndarray):
        res = res[0]
    
    return res

def train_with_obs(model, stimuli, rewards, actions, fixed=None):

    if fixed:
        model.paras.update(fixed)

    x0 = list(model.paras.values())

    def objective_func(x0):
        return - loglike(model, stimuli, rewards, actions)

    opt_results = minimize(fun=objective_func, x0=x0) 

    return opt_results

def simulate(env, model, n_trials, seed=0):
    """Simulation in a given AI Gym environment."""
    assert isinstance(env, gym.Env)
    assert isinstance(model, sciunit.Model)
    assert isinstance(n_trials, int)
    
    # reset the agent state and 
    model.reset()
    init_stimulus = env.reset()

    actions = []
    rewards = [] # np.zeros(n_trials, dtype=int)
    stimuli = [] # np.zeros(n_trials, dtype=int)

    # add the first stimulus in the environment
    stimuli.append(init_stimulus)
    # stimuli = np.insert(stimuli, 0, init_stimulus, axis=0)

    for i in range(n_trials):
        # action based on choice probabilities
        # actions[i] = model.act(stimuli[i])
        s = stimuli[-1]
        a = model.act(s)
        actions.append(a)

        # generate reward based on action
        s_update, r, done, _ = env.step(a)
        rewards.append(r)
        stimuli.append(s_update)

        # update choice kernel and Q weights
        model.update(s, r, a, done)

    # delete the extra stimulus
    # stimuli = np.delete(stimuli, n_trials, axis=0)

    env.close()

    return stimuli[1:], rewards, actions

class MultiMeta(type):
    def __new__(cls, name, bases, dct):
        single_cls = dct['single_cls']
        base_classes = (single_cls.__bases__)
        out_cls = super().__new__(cls, name, base_classes, dct)

        def multi_init(self, param_list, *args, **kwargs):
            self.models = []
            for param in param_list:
                self.models.append(single_cls(*args, **kwargs, paras=param))
        out_cls.__init__ = multi_init

        def multi_predict(self, idx, *args, **kwargs):
            return self.models[idx].predict(*args, **kwargs)
        out_cls.predict = multi_predict

        def multi_update(self, idx, *args, **kwargs):
            return self.models[idx].update(*args, **kwargs)
        out_cls.update = multi_update

        def multi_act(self, idx, *args, **kwargs):
            return self.models[idx].act(*args, **kwargs)
        out_cls.act = multi_act

        def multi_reset(self, idx, *args, **kwargs):
            return self.models[idx].reset(*args, **kwargs)
        out_cls.reset = multi_reset



        return out_cls

def multi_from_single(single_cls, multi_cls_name):
    return MultiMeta(multi_cls_name, (), {'single_cls': single_cls})