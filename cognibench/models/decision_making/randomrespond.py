import numpy as np
from gym import spaces
from scipy import stats

from cognibench.models import CNBAgent
from cognibench.models.policy_model import PolicyModel
from cognibench.capabilities import Interactive, PredictsLogpdf
from cognibench.capabilities import (
    ProducesPolicy,
    DiscreteAction,
    DiscreteObservation,
)
from overrides import overrides


class RandomRespondAgent(CNBAgent, ProducesPolicy, DiscreteAction, DiscreteObservation):
    """
    Random respond agent that performs random actions for any kind of stimulus.
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
            bias : float
                Bias probability. Must be in range [0, 1].

            action_bias : int
                ID of the action. Must be in range [0, n_action)
        """
        self.set_action_space(n_action)
        self.set_observation_space(n_obs)
        super().__init__(*args, **kwargs)

    def reset(self):
        """
        Override base class reset behaviour by setting the hidden state to default values.
        """
        self.set_hidden_state(dict())

    def eval_policy(self, stimulus):
        """
        Return a random variable object from the given stimulus.
        """
        assert self.get_observation_space().contains(stimulus)

        bias = self.get_paras()["bias"]
        action_bias = int(self.get_paras()["action_bias"])

        n = self.n_action()
        pk = np.full(n, (1 - bias) / (n - 1))
        pk[action_bias] = bias

        xk = np.arange(n)
        rv = stats.rv_discrete(values=(xk, pk))
        rv.random_state = self.get_seed()

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
        Doesn't do anything. Stimulus and action must be from their respective
        spaces.
        """
        assert self.get_action_space().contains(action)
        assert self.get_observation_space().contains(stimulus)


class RandomRespondModel(PolicyModel, DiscreteAction, DiscreteObservation):
    """
    Random respond model implementation.
    """

    name = "RandomRespondModel"

    @overrides
    def __init__(self, *args, n_action, n_obs, seed=None, **kwargs):
        self.set_action_space(n_action)
        self.set_observation_space(n_obs)
        agent = RandomRespondAgent(n_action=n_action, n_obs=n_obs, seed=seed)

        def initializer(seed):
            return {
                "bias": stats.uniform.rvs(loc=0, scale=1, random_state=seed),
                "action_bias": int(
                    stats.uniform.rvs(loc=0, scale=n_action, random_state=seed)
                ),
            }

        super().__init__(
            *args, agent=agent, param_initializer=initializer, seed=seed, **kwargs
        )
