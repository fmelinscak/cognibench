import sciunit
import numpy as np
from gym.utils import seeding
from overrides import overrides


class LDMModel(sciunit.Model):
    """
    Helper base class for LDMUnit models.
    """

    @overrides
    def __init__(self, seed=None, param_initializer=None, **kwargs):
        """
        Parameters
        ----------
        seed : int
            Random seed. Must be a nonnegative integer. If seed is None,
            random state is set randomly by gym.utils.seeding. (Default: None)
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
        a distribution over actions, but the exact return type would depend on how `ldmunit` library is being used.
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
        if self.param_initializer is None:
            raise ValueError(
                "{self.name.init_params}: self.param_initializer is None; cannot initializer parameters"
            )
        self.set_paras(self.param_initializer(self.seed))

    def set_paras(self, *args, **kwargs):
        """
        Set the parameters for the model.
        """
        pass

    def get_paras(self):
        """
        Get the parameters of the model.
        """
        return dict()


class LDMAgent:
    def __init__(self, *args, paras_dict=None, seed=None, **kwargs):
        self.seed = seed
        self.paras = paras_dict
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
        raise NotImplementedError("LDMAgent must implement act")

    def update(self, *args, **kwargs):
        raise NotImplementedError("LDMAgent must implement update")

    def reset(self):
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
