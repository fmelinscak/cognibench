import numpy as np
from gym import spaces
from scipy import stats

from ldmunit.models import LDMAgent
from ldmunit.models.policy_model import PolicyModel
from ldmunit.capabilities import Interactive, PredictsLogpdf
from ldmunit.capabilities import (
    ProducesPolicy,
    DiscreteAction,
    DiscreteObservation,
)
from overrides import overrides


class NWSLSAgent(LDMAgent, ProducesPolicy, DiscreteAction, DiscreteObservation):
    """
    Noisy-win-stay-lose-shift agent implementation.
    """

    @overrides
    def __init__(self, *args, n_action, n_obs, **kwargs):
        """
        Parameters
        ----------
        n_action : int
            Dimension of the action space.

        n_obs : int
            Dimension of the observation space.

        paras_dict : dict (optional)
            epsilon : int
                Number of loose actions. Must be nonnegative and less than or equal
                to the dimension of the action space.
        """
        self.set_action_space(n_action)
        self.set_observation_space(n_obs)
        super().__init__(*args, **kwargs)

    def reset(self):
        """
        Override base class reset behaviour by setting the hidden state to default
        values for NWSLS model.
        """
        self.hidden_state = dict(win=True, action=int(self.paras["epsilon"]))

    def eval_policy(self, stimulus):
        """
        Return a random variable object from the given stimulus.
        """
        assert self.get_observation_space().contains(stimulus)

        epsilon = self.paras["epsilon"]
        n = self.n_action()

        if self.hidden_state["win"]:
            prob_action = 1 - epsilon / n
        else:
            prob_action = epsilon / n

        pk = np.full(n, (1 - prob_action) / (n - 1))
        pk[self.hidden_state["action"]] = prob_action

        xk = np.arange(n)
        rv = stats.rv_discrete(name=None, values=(xk, pk))
        rv.random_state = self.seed

        return rv

    def act(self, stimulus):
        """
        Return an action for the given stimulus.

        Parameters
        ----------
        stimulus : int
            A stimulus from the observation space for this model.

        Returns
        -------
        int
            An action from the action space.
        """
        return self.eval_policy(stimulus).rvs()

    def update(self, stimulus, reward, action, done=False):
        """
        Update the hidden state of the model based on input stimulus, action performed
        by the model and reward.

        Parameters
        ----------
        stimulus : int
            A stimulus from the observation space for this model.

        reward : int
            The reward for the action.

        action : int
            Action performed by the model.
        """
        assert self.get_action_space().contains(action)
        assert self.get_observation_space().contains(stimulus)

        self.hidden_state["win"] = reward == 1
        self.hidden_state["action"] = action


class NWSLSModel(PolicyModel):
    """
    Noisy-win-stay-lose-shift model implementation.
    """

    name = "NWSLSModel"

    @overrides
    def __init__(self, *args, n_action, n_obs, seed=None, **kwargs):
        agent = NWSLSAgent(n_action=n_action, n_obs=n_obs)

        def initializer(seed):
            return {
                "epsilon": int(
                    stats.uniform.rvs(loc=0, scale=n_action, random_state=seed)
                )
            }

        super().__init__(
            *args, agent=agent, param_initializer=initializer, seed=seed, **kwargs
        )
