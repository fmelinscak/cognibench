import sciunit
from gym.utils import seeding
from collections.abc import Mapping


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
