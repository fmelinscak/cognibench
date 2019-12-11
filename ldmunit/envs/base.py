import gym
from gym.utils import seeding


class LDMEnv(gym.Env):
    name = "LDMEnv"

    def __init__(self, *args, seed=None, **kwargs):
        """
        Parameters
        ----------
        seed : int
            Random seed to use
        """
        self.seed(seed)
        super().__init__(*args, **kwargs)

    def seed(self, seed=None):
        """Set the random_state for the environment if given.

        Parameters
        ----------
        seed : int
            Seed for the random_state
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def update(self, stimulus, reward, action, done=False):
        """
        Method to update the internal state of the environment. If you have a stateful
        environment, override this method to specify the state transitions.
        """
        pass
