import gym


class LDMEnv(gym.Env):
    def update(self, stimulus, reward, action, done=False):
        """
        Method to update the internal state of the environment. If you have a stateful
        environment, override this method to specify the state transitions.
        """
        pass
