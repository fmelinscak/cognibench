import numpy as np
from scipy.special import softmax
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


class RWCKAgent(LDMAgent, ProducesPolicy, DiscreteAction, DiscreteObservation):
    """
    Rescorla-Wagner choice kernel agent implementation.
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
            w : float
                Initial value of every element of weight matrix Q.

            beta : float
                Multiplicative factor used to multiply a row of Q matrix when computing
                logits.

            beta_c : float
                Multiplicative factor used to multiply a row of CK matrix when computing
                logits.

            eta : float
                Learning rate for Q updates. Must be nonnegative.

            eta_c : float
                Learning rate for CK updates. Must be nonnegative.

        Other Parameters
        ----------------
        **kwargs : any type
            All the mandatory keyword-only arguments required by :class:`ldmunit.models.decision_making.base.DADO` must also be
            provided during initialization.
        """
        self.set_action_space(n_action)
        self.set_observation_space(n_obs)
        super().__init__(*args, **kwargs)

    def reset(self):
        w = self.get_paras()["w"]
        self.set_hidden_state(
            {
                "CK": np.zeros((self.n_obs(), self.n_action())),
                "Q": np.full((self.n_obs(), self.n_action()), w),
            }
        )

    def eval_policy(self, stimulus):
        """
        Return a random variable object from the given stimulus.
        """
        assert self.get_observation_space().contains(stimulus)
        CK_i = self.get_hidden_state()["CK"][stimulus]
        Q_i = self.get_hidden_state()["Q"][stimulus]

        beta = self.get_paras()["beta"]
        beta_c = self.get_paras()["beta_c"]
        V = beta * Q_i + beta_c * CK_i

        xk = np.arange(self.n_action())
        pk = softmax(V)
        rv = stats.rv_discrete(values=(xk, pk))
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

        done : bool
            If `True`, do not do any update.
        """
        assert self.get_action_space().contains(action)
        assert self.get_observation_space().contains(stimulus)

        # get model's state
        CK, Q = (
            self.get_hidden_state()["CK"][stimulus],
            self.get_hidden_state()["Q"][stimulus],
        )

        if not done:
            # unpack parameters
            eta = self.get_paras()["eta"]
            eta_c = self.get_paras()["eta_c"]

            # update choice kernel
            CK = (1 - eta_c) * CK
            CK[action] += eta_c

            # update Q weights
            delta = reward - Q[action]
            Q[action] += eta * delta

            self.get_hidden_state()["CK"][stimulus] = CK
            self.get_hidden_state()["Q"][stimulus] = Q

        return CK, Q


class RWCKModel(PolicyModel, DiscreteAction, DiscreteObservation):
    """
    Rescorla-Wagner choice kernel model implementation using RWCK agent implementation as the underlying agent equations.
    """

    name = "RWCKModel"

    @overrides
    def __init__(self, *args, n_action, n_obs, seed=None, **kwargs):
        self.set_action_space(n_action)
        self.set_observation_space(n_obs)
        agent = RWCKAgent(n_action=n_action, n_obs=n_obs)

        def initializer(seed):
            return {
                "w": stats.uniform.rvs(scale=5, random_state=seed),
                "beta": stats.uniform.rvs(scale=5, random_state=seed),
                "beta_c": stats.uniform.rvs(scale=5, random_state=seed),
                "eta": 1e-2,
                "eta_c": 1e-2,
            }

        super().__init__(
            *args, agent=agent, param_initializer=initializer, seed=seed, **kwargs
        )


class RWModel(PolicyModel, DiscreteAction, DiscreteObservation):
    """
    Rescorla-Wagner model implementation as a special case of Rescorla-Wagner choice kernel model.
    """

    name = "RWModel"

    @overrides
    def __init__(self, *args, n_action, n_obs, seed=None, **kwargs):
        self.set_action_space(n_action)
        self.set_observation_space(n_obs)
        agent = RWCKAgent(n_action=n_action, n_obs=n_obs)

        def initializer(seed):
            return {
                "w": stats.uniform.rvs(scale=5, random_state=seed),
                "beta": stats.uniform.rvs(scale=5, random_state=seed),
                "beta_c": 0,
                "eta": 1e-2,
                "eta_c": 1e-2,
            }

        super().__init__(
            *args, agent=agent, param_initializer=initializer, seed=seed, **kwargs
        )


class CKModel(PolicyModel, DiscreteAction, DiscreteObservation):
    """
    Choice kernel model implementation as a special case of Rescorla-Wagner choice kernel model.
    """

    name = "CKModel"

    @overrides
    def __init__(self, *args, n_action, n_obs, seed=None, **kwargs):
        self.set_action_space(n_action)
        self.set_observation_space(n_obs)
        agent = RWCKAgent(n_action=n_action, n_obs=n_obs)

        def initializer(seed):
            return {
                "w": stats.uniform.rvs(scale=5, random_state=seed),
                "beta": 0,
                "beta_c": stats.uniform.rvs(scale=5, random_state=seed),
                "eta": 1e-2,
                "eta_c": 1e-2,
            }

        super().__init__(
            *args, agent=agent, param_initializer=initializer, seed=seed, **kwargs
        )
