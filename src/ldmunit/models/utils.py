from scipy.optimize import minimize
import gym
import sciunit
import numpy as np

def loglike(model, stimuli, rewards, actions):
    
    res = 0
    model.reset()

    for s, r, a in zip(stimuli, rewards, actions):
        log_prob = model.predict(s)
        res += log_prob(a)
        model.update(s, r, a, False)

    return res

def train_with_obs(model, stimuli, rewards, actions, paras_x0):

    if not model.hidden_state:
        model.set_space_from_data(stimuli, actions)
        model.reset()

    bounds = []
    x0 = []

    for k, v in paras_x0.items():
        assert k in model.paras, "Supplied parameter is not in the model's parameters' list"
        bounds.append(v)
        lb, ub = v
        start = np.random.uniform(lb, ub)
        x0.append(start)
        # model.paras[k] = start
    
    def objective_func(x0):
        model.paras.update(dict(zip(paras_x0.keys(), [0.12])))
        return - loglike(model, stimuli, rewards, actions)

    opt_results = minimize(fun=objective_func, x0=x0, bounds=bounds, method='L-BFGS-B')

    return opt_results

def simulate_single_env_single_model(env, model, n_trials, seed=0):
    assert isinstance(env, gym.Env)
    assert isinstance(model, sciunit.Model)
    
    s_init = env.reset()
    env.seed(seed)
    actions = []
    rewards = []
    stimuli = []

    stimuli.append(s_init)

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

def simulate_multi_env_single_model(env, model, n_trials, seed=0):
    """Simulation in a given AI Gym environment."""
    model.reset()

    if isinstance(env, list):
        actions = []
        rewards = []
        stimuli = []
        tmp = np.linspace(0,n_trials, len(env)+1, dtype=int)
        n_trials_list = tmp[1:] - tmp[:-1]
        for i in range(len(env)):
            n = n_trials_list[i]
            assert env[i].observation_space == model.observation_space, "Observation space must be the same between environment and the model"
            assert env[i].action_space == model.action_space, "Action space must be the same between environment and the model"
            s, r, a = simulate_single_env_single_model(env[i], model, n, seed)
            stimuli.extend(s)
            rewards.extend(r)
            actions.extend(a)
    else:
        assert env.observation_space == model.observation_space, "Observation space must be the same between environment and the model"
        assert env.action_space == model.action_space, "Action space must be the same between environment and the model"
        stimuli, rewards, actions = simulate_single_env_single_model(env, model, n_trials, seed)
    
    return stimuli, rewards, actions

def simulate(env, model, n_trials, seed=0):
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
            paras = self.models[idx].paras
            return self.models[idx].predict(paras=paras, *args, **kwargs)
        out_cls.predict = multi_predict

        def multi_update(self, idx, *args, **kwargs):
            paras = self.models[idx].paras
            return self.models[idx].update(paras=paras, *args, **kwargs)
        out_cls.update = multi_update

        def multi_act(self, idx, *args, **kwargs):
            paras = self.models[idx].paras
            return self.models[idx].act(paras=paras, *args, **kwargs)
        out_cls.act = multi_act

        def multi_reset(self, idx, *args, **kwargs):
            paras = self.models[idx].paras
            return self.models[idx].reset(paras=paras, *args, **kwargs)
        out_cls.reset = multi_reset

        return out_cls

def multi_from_single(single_cls):
    multi_cls_name = 'Multi' + single_cls.name
    return MultiMeta(multi_cls_name, (), {'single_cls': single_cls})