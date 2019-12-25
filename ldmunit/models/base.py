import sciunit
import numpy as np
from gym.utils import seeding
from overrides import overrides
from collections import Mapping


class LDMModel(sciunit.Model):
    """
    Base class for LDMUnit models.

    In `ldmunit`, a model is a way of representing a continuum of some type of an agent and corresponding possible parameters. Models are
    expected to provide fitting and action prediction functionalities, while leaving the tasks of acting on environments and
    updating hidden state variables to agents.
    """

    def __init__(self, seed=None, param_initializer=None, **kwargs):
        """
        Parameters
        ----------
        seed : int
            Random seed. Must be a nonnegative integer. If seed is None,
            random state is set randomly by gym.utils.seeding. (Default: None)

        param_initializer : dict or callable
            Initializer for model parameters. If a dictionary, it must be a mapping from model parameters to initial
            values. If a callable, it should have the below signature

                `param_initializer(seed=None) -> dict`

            and must return a mapping from model parameters to initial values.
        """
        self.seed = seed
        self.param_initializer = param_initializer
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

    def fit(self, *args, **kwargs):
        """
        Fit the model to a batch of stimuli. If this is a multi-subject model, then the stimuli should be a list
        where each element contains the stimuli of the corresponding subject.

        By default, this method does not perform model fitting. This method should be overridden only if you need
        model fitting functionality.
        """
        pass

    def predict(self, *args, **kwargs):
        """
        Make a prediction over the action space given a stimulus. `predict` function is generally expected to return
        a distribution over actions.
        """
        raise NotImplementedError("Must implement predict.")

    def act(self, *args, **kwargs):
        """
        Return an action given a stimulus.
        """
        raise NotImplementedError("Must implement act.")

    def reset(self):
        """
        Reset the hidden state of the model. Subclasses should override
        this method with suitable default hidden state values so that hidden
        state is set to this default during object initialization.
        """
        pass

    def init_paras(self):
        """
        Initialize model parameters using `self.param_initializer`.
        """
        if self.param_initializer is None:
            raise ValueError(
                "{self.name.init_params}: self.param_initializer is None; cannot initializer parameters"
            )
        if isinstance(self.param_initializer, Mapping):
            paras = self.param_initializer
        else:
            paras = self.param_initializer(seed=self.seed)
        self.set_paras(paras)

    def set_paras(self, paras_dict):
        """
        Set the parameters for the model.
        """
        self._paras = paras_dict

    def get_paras(self):
        """
        Get the parameters of the model.
        """
        return self._paras


class LDMAgent:
    """
    Base class for LDMUnit agents.

    In `ldmunit`, an agent is a way of interacting with environments through `act` and `update` methods while possibly
    storing some hidden state.
    """

    def __init__(self, *args, paras_dict=None, seed=None, **kwargs):
        """
        Parameters
        ----------
        paras_dict : dict
            Dictionary storing agent parameters.

        seed : int
            Random seed to use.
        """
        self.paras = paras_dict
        self.seed = seed
        self.hidden_state = dict()
        super().__init__(*args, **kwargs)

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

    def act(self, *args, **kwargs):
        """
        Act on the stimulus given by the environment.
        """
        raise NotImplementedError("LDMAgent must implement act")

    def update(self, *args, **kwargs):
        """
        Update hidden state using the information provided by previous stimulus, action taken and reward returned by the
        environment.
        """
        raise NotImplementedError("LDMAgent must implement update")

    def reset(self):
        """
        Reset the hidden state of the agent.

        Concrete classes should override this method and define default hidden state variables.
        """
        self.hidden_state = dict()

    @property
    def paras(self):
        return self._paras

    @paras.setter
    def paras(self, paras_dict):
        self._paras = paras_dict

    @property
    def hidden_state(self):
        return self._hidden_state

    @hidden_state.setter
    def hidden_state(self, state):
        self._hidden_state = state
