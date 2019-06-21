def loglike(model, stimuli, rewards, actions):
    
    res = 0
    model.reset()

    for s, r, a in zip(stimuli, rewards, actions):
        # compute choice probabilities
        pmf = model.predict(s)
        
        # probability of the action
        p = P[a

        # add log-likelihood
        res += np.log(p)
        # update choice kernel and Q weights
        model.update(s, r, a, False)
    
    return res

def train_with_obs(model, stimuli, rewards, actions, fixed):

    x0 = list(fixed.values())

    def objective_func(x0):
        for k, v in zip(fixed.keys(), x0):
            model.paras[k] = v
        return - loglike(model, stimuli, rewards, actions)

    opt_results = minimize(fun=objective_func, x0=x0) 

    return opt_results