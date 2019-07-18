import numpy as np
import gym

class ContinuousSpace(gym.Space):
    """
    Continuous space for numbers in :math:`\mathbb{R}`.
    """
    def __init__(self):
        """
        Initialize the space using float64 :class:`numpy.dtype`
        """
        super().__init__((), np.float64)

    def sample(self):
        pass

    def contains(self, x):
        """
        Assert if a value is in :math:`\mathbb{R}` space. Integers and floating numbers
        (including :class:`numpy.ndarray`) will also be included.

        Parameters
        ----------
        x : int or float or :class:`numpy.ndarray`
            Value to check

        Returns
        -------
        bool
        """
        if isinstance(x, float):
            as_float = x
        elif isinstance(x, (np.generic, np.ndarray)) and (x.dtype.kind in np.typecodes['AllFloat'] and x.shape == ()):
            as_float = float(x)
        # ensure compatiable with int
        elif isinstance(x, int):
            as_float = float(x)
        elif isinstance(x, (np.generic, np.ndarray)) and (x.dtype.kind in np.typecodes['AllInteger'] and x.shape == ()):
            as_float = float(x)
        else:
            return False
        return isinstance(as_float, float)

    def __repr__(self):
        return "Continuous"

    def __eq__(self, other):
        return isinstance(other, ContinuousSpace)
