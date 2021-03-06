import numpy as np
from gym import spaces
from scipy import stats

from cognibench.distr import DiscreteRV
from cognibench.models import CNBAgent
from cognibench.models.policy_model import PolicyModel
from cognibench.capabilities import Interactive, PredictsLogpdf
from cognibench.capabilities import (
    ProducesPolicy,
    DiscreteAction,
    DiscreteObservation,
)
from overrides import overrides


class NWSLSAgent(CNBAgent, ProducesPolicy, DiscreteAction, DiscreteObservation):
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
            epsilon : float
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
        self.set_hidden_state(
            dict(
                win=True,
                action=min(int(self.get_paras()["epsilon"]), self.n_action() - 1),
            )
        )

    def eval_policy(self, stimulus):
        """
        Return a random variable object from the given stimulus.
        """
        assert self.get_observation_space().contains(stimulus)

        epsilon = self.get_paras()["epsilon"]
        n = self.n_action()

        a = self.get_hidden_state()["action"]
        if self.get_hidden_state()["win"]:
            pk = np.full(n, epsilon / n)
            pk[a] = 1 - (n - 1) * epsilon / n
        else:
            if n == 1:
                pk[0] = 1
            else:
                pk = np.full(n, (1 - epsilon / n) / (n - 1))
                pk[a] = epsilon / n

        rv = DiscreteRV(pk)
        rv.random_state = self.rng
        # xk = np.arange(n)
        # rv = stats.rv_discrete(name=None, values=(xk, pk))
        # rv.random_state = self.rng

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

        self.get_hidden_state()["win"] = reward == 1
        self.get_hidden_state()["action"] = action


class NWSLSModel(PolicyModel, DiscreteAction, DiscreteObservation):
    """
    Noisy-win-stay-lose-shift model implementation.
    """

    name = "NWSLSModel"

    @overrides
    def __init__(self, *args, n_action, n_obs, seed=None, **kwargs):
        self.set_action_space(n_action)
        self.set_observation_space(n_obs)
        agent = NWSLSAgent(n_action=n_action, n_obs=n_obs, seed=seed)

        def initializer(seed):
            return {
                "epsilon": stats.uniform.rvs(loc=0, scale=n_action, random_state=seed)
            }

        self.param_bounds = {"epsilon": (0, n_action)}
        super().__init__(
            *args, agent=agent, param_initializer=initializer, seed=seed, **kwargs
        )
