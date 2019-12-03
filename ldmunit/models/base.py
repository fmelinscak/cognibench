import sciunit
import numpy as np
from gym.utils import seeding
from collections.abc import Mapping
from gym import spaces
from ldmunit.capabilities import DiscreteAction, DiscreteObservation
from ldmunit.capabilities import (
    MultiBinaryObservation,
    ContinuousAction,
    ContinuousObservation,
    ReturnsNumParams,
)
from ldmunit.continuous import ContinuousSpace


class LDMModel(sciunit.Model):
    """
    Helper base class for LDMUnit models.
    """

    def __init__(self, paras=None, hidden_state=None, seed=None, **kwargs):
        """
        Parameters
        ----------
        paras : dict
            Model parameters. (Default: empty dict)

        hidden_state : dict
            Hidden state of the model. (Default: empty dict)

        seed : int
            Random seed. Must be a nonnegative integer. If seed is None,
            random state is set randomly by gym.utils.seeding. (Default: None)
        """
        self.seed = seed
        self.paras = paras
        if hidden_state is None:
            self.reset()
        else:
            self.hidden_state = hidden_state
        super().__init__(**kwargs)

    @property
    def seed(self):
        """
        Returns
        -------
        int or None
            Random seed used to initialize the random number generator.
            Seed is None only if it was omitted during model initialization.
        """
        return self._seed

    @property
    def rng(self):
        """
        Returns
        -------
        :class:`numpy.random.RandomState`
            Random number generator state. Use this object as an np.random
            replacement to generate random numbers. This way, you can reproduce
            your results if you always use the same seed during model initialization.
        """
        return self._rng

    @seed.setter
    def seed(self, value):
        self._seed = value
        self._rng, _ = seeding.np_random(seed=value)

    @property
    def hidden_state(self):
        return self._hidden_state

    def predict(self, *args, **kwargs):
        """
        Make a prediction given a stimuli.
        """
        raise NotImplementedError("Must implement predict.")

    def act(self, *args, **kwargs):
        """
        For decision making, return the action taken by the model.
        Associative learning models should return the predicted value.
        Also named observation function in some packages.
        """
        raise NotImplementedError("Must implement act")

    def reset(self):
        """
        Reset the hidden state of the model. Subclasses should override
        this method with suitable default hidden state values so that hidden
        state is set to this default during object initialization.
        """
        self.hidden_state = dict()

    @hidden_state.setter
    def hidden_state(self, value):
        if value is None:
            self._hidden_state = dict()
        elif not isinstance(value, Mapping):
            raise TypeError("hidden_state must be of dict type")
        else:
            self._hidden_state = value

    @property
    def paras(self):
        return self._paras

    @paras.setter
    def paras(self, value):
        if value is None:
            self._paras = dict()
        elif not isinstance(value, Mapping):
            raise TypeError("paras must be of dict type")
        else:
            self._paras = value


class DADO(LDMModel, DiscreteAction, DiscreteObservation):
    """
    Base class for models that operate on discrete action and discrete observation spaces.
    """

    def __init__(self, *args, n_action, n_obs, **kwargs):
        """
        Parameters
        ----------
        n_action : int
            Dimension of the action space.

        n_obs : int
            Dimension of the observation space.
        """
        assert n_action > 0, "n_action must be positive"
        assert n_obs > 0, "n_obs must be positive"
        self.action_space = n_action
        self.observation_space = n_obs
        super().__init__(**kwargs)

    def set_space_from_data(self, stimuli, actions):
        """
        Infer action and observation spaces from given simulation data.

        Parameters
        ----------
        stimuli : array-like
            Each element of stimuli must contain one stimulus. Further, each stimulus
            be an array-like object whose length will be used as the dimension of the
            observation space.

        actions : array-like
            Each element of actions must contain one action. Further, each action
            be an array-like object whose length will be used as the dimension of the
            action space.
        """
        assert self._check_observation(stimuli) and self._check_action(actions)
        if not len(stimuli) == len(actions):
            raise AssertionError("stimuli and actions must be of the same length.")
        self.action_space = len(np.unique(actions))
        self.observation_space = len(np.unique(stimuli))
        print("action_space set to {}".format(self.action_space))
        print("observation_space set to {}".format(self.observation_space))


class CACO(LDMModel, ContinuousAction, ContinuousObservation):
    """
    Base class for models that operate on continuous action and continuous observation spaces.
    """

    def __init__(self, *args, **kwargs):
        self.action_space = ContinuousSpace()
        self.observation_space = ContinuousSpace()
        super().__init__(*args, **kwargs)


class CAMO(LDMModel, ContinuousAction, MultiBinaryObservation):
    """
    Base class for models that operate on continuous action and multi-binary observation spaces.
    """

    def __init__(self, *args, n_obs, **kwargs):
        """
        Parameters
        ----------
        n_obs : int
            Dimension of the multi-binary observation space. For example, when n_obs is 4,
            [0, 1, 1, 0] is a possible sample from the observation space since it consists of
            4 binary values. Must be positive.
        """
        assert n_obs > 0, "n_obs must be positive"
        self.action_space = ContinuousSpace()
        self.observation_space = n_obs
        super().__init__(**kwargs)

    def set_space_from_data(self, stimuli, actions):
        """
        Infer action and observation spaces from given simulation data.

        Parameters
        ----------
        stimuli : array-like
            Each element of stimuli must contain one stimulus. Further, each stimulus
            be an array-like object whose length will be used as the dimension of the
            observation space.

        actions : array-like
            Each element of actions must contain a continuous action. Lengths of
            stimuli and actions must be the same.
        """
        assert self._check_observation(stimuli) and self._check_action(actions)
        if not len(stimuli) == len(actions):
            raise AssertionError("stimuli and actions must be of the same length.")
        self.action_space = ContinuousSpace()
        self.observation_space = len(stimuli[0])


class ParametricModelMixin(ReturnsNumParams):
    """
    A simple mixin class that allows easy ReturnsNumParams interface implementation
    for parametric models. It is assumed that the deriving class has a sequence or dictionary
    field `self.paras` which stores all the parameters of the model separately. For more
    sophisticated models, implementing the ReturnsNumParams interface yourself may be easier and
    more accurate.
    """

    def n_params(self):
        return len(self.paras)
