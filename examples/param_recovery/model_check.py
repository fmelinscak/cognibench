import numpy as np
import scipy.stats as stats
from scipy.special import logit, expit

from os import getcwd
import sciunit
from cognibench.tasks import param_recovery
from cognibench.models.decision_making import RandomRespondModel
from cognibench.testing import InteractiveTest
from cognibench.envs import BanditEnv
from cognibench.utils import partialclass
from cognibench import simulation
import matplotlib.pyplot as plt

sciunit.settings["CWD"] = getcwd()


paras_list = []
true = []
est = []
for i in range(101, 201):
    print(f"Fit {i}")
    model = RandomRespondModel(n_action=2, n_obs=2, seed=i)
    env = BanditEnv(p_dist=[0.2, 0.8])
    seed = i + int(1e5)
    rng = np.random.RandomState(seed)
    sim_paras = {"logit": logit(rng.uniform()), "action_bias": 0}
    model.set_paras(sim_paras)
    model.reset()
    stimuli, rewards, actions = simulation.simulate(env, model, 1000)
    model.init_paras()
    model.reset()
    model.fit(stimuli, rewards, actions)
    true.append(expit(sim_paras["logit"]))
    est.append(expit(model.get_paras()["logit"]))

plt.scatter(true, est)
plt.show()
