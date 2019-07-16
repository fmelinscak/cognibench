import sciunit
from gym.utils import seeding


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
            Random seed. Must be a nonnegative integer. (Default: None)
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
        return self._seed

    @property
    def rng(self):
        return self._rng

    @seed.setter
    def seed(self, value):
        self._rng, self._seed = seeding.np_random(seed=seeding.create_seed(max_bytes=4))

    @property
    def hidden_state(self):
        return self._hidden_state

    def reset(self):
        self.hidden_state = dict()

    @hidden_state.setter
    def hidden_state(self, value):
        if value is None:
            self._hidden_state = dict()
        elif not isinstance(value, dict):
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
        elif not isinstance(value, dict):
            raise TypeError("paras must be of dict type")
        else:
            self._paras = value
