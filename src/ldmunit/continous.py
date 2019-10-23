import numpy as np
import gym

class Continous(gym.Space):
    """Continous space for numbers in R^1.
    """
    def __init__(self):
        super().__init__((), np.float64)

    def sample(self):
        pass

    def contains(self, x):
        """Assert if a value is in R^1 space.
        
        Integers and floating numbers (including np.ndarray) will also be included.
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
        return "Continous"

    def __eq__(self, other):
        return isinstance(other, Continous)