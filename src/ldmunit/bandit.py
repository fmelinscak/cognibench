import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
from .continous import Continous

class BanditEnv(gym.Env):
    """
    Bandit environment base to allow agents to interact with the class n-armed bandit
    in different variations

    p_dist:
        A list of probabilities of the likelihood that a particular bandit will pay out
    info:
        Info about the environment that the agents is not supposed to know. For instance,
        info can releal the index of the optimal arm, or the value of prior parameter.
        Can be useful to evaluate the agent's perfomance
    """
    def __init__(self, p_dist, info={}):
        if min(p_dist) < 0 or max(p_dist) > 1:
            raise ValueError("All probabilities must be between 0 and 1")

        self.p_dist = p_dist

        self.info = info

        self.n_bandits = len(p_dist)
        self.action_space = spaces.Discrete(self.n_bandits)
        self.observation_space = spaces.Discrete(1)
        self.action_space = spaces.Box(-1000, 1000, shape=(1,), dtype=np.float32)

        
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):

        reward = 0
        done = False

        if np.random.uniform() < self.p_dist[action]:
            reward = 1

        return 0, reward, done, self.info

    def reset(self):
        return np.random.randint(0, self.n_bandits)

    def render(self, mode='human', close=False):
        pass

class BanditAssociateEnv(gym.Env):
    def __init__(self, p_dist, info={}):

        if min(p_dist) < 0 or max(p_dist) > 1:
            raise ValueError("All probabilities must be between 0 and 1")

        self.p_dist = p_dist
        self.info = info

        self.n_bandits = len(p_dist)
        self.observation_space = spaces.MultiBinary(self.n_bandits)
        self.action_space = Continous()

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action)

        reward = 0
        done = False

        reward = np.random.randint(0, 1)
        observation = self.observation_space.sample()

        return observation, reward, done, self.info

    def reset(self):
        return self.observation_space.sample()

    def render(self, mode='human', close=False):
        pass