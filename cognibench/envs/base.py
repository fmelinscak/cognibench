import gym
from gym.utils import seeding


class CNBEnv(gym.Env):
    """
    Environment class that build upon :class:`gym.Env`. In addition to functionality provided by :class:`gym.Env`,
    `CNBEnv` allows the environment to be stateful, and this state to be updated using :py:meth:`CNBEnv.update` method.
    """

    name = "CNBEnv"

    def __init__(self, *args, seed=None, **kwargs):
        """
        Parameters
        ----------
        seed : int
            Random seed to use
        """
        self.set_seed(seed)
        super().__init__(*args, **kwargs)

    def set_seed(self, seed):
        """Set the random_state for the environment if given.

        Parameters
        ----------
        seed : int
            Seed for the random_state
        """
        self.np_random, seed = seeding.np_random(seed)
        return seed

    def update(self, stimulus, reward, action, done=False):
        """
        Method to update the internal state of the environment. If you have a stateful environment, override this method
        to specify the state transitions. By default, environments are defined to be stateless (hence this method does
        nothing).
        """
        pass
