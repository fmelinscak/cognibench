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
        self.observation_space = spaces.Discrete(self.n_bandits)
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):

        reward = 0
        done = False

        if self.np_random.uniform() < self.p_dist[action]:
            reward = 1

        return 0, reward, done, self.info

    def reset(self):
        return self.np_random.randint(0, self.n_bandits)

    def render(self, mode='human', close=False):
        pass

class BanditAssociateEnv(gym.Env):
    """
    Environment base to allow agents to learn from stimulus occuring at different
    probabilities.

    stimuli:
        A list of stimulus in the same gym.spaces.MultiBinary space.

    p_stimuli:
        A list of probabilities that a stimulus will occur

    p_reward:
        A list of probabilities of the likelihood that a particular stimuli will pay out
    info:
        Info about the environment that the agents is not supposed to know. For instance,
        info can releal the index of the optimal arm, or the value of prior parameter.
        Can be useful to evaluate the agent's perfomance
    """
    def __init__(self, stimuli, p_stimuli, p_reward, info={}):

        if min(p_stimuli) < 0 or max(p_stimuli) > 1 or sum(p_stimuli) != 1:
            raise ValueError("All probabilities must be between 0 and 1")
        if min(p_reward) < 0 or max(p_reward) > 1:
            raise ValueError("All probabilities must be between 0 and 1")

        assert len(set(map(len, (p_stimuli, stimuli, p_reward)))) == 1, "Stimuli and Probability list must be of equal length"
        self._n = len(stimuli[0])
        self.observation_space = spaces.MultiBinary(self._n)
        self.action_space = Continous()
        for s in stimuli:
            assert self.observation_space.contains(s), "Stimuli must be in the same MultiBinary space"

        self.stimuli = stimuli
        self.p_stimuli = p_stimuli
        self.p_reward = p_reward
        self.info = info


        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action)

        reward = 0
        done = False

        obs_idx = self.np_random.choice(range(len(self.stimuli)), 
                                   p=self.p_stimuli, replace=True)

        if self.np_random.uniform() < self.p_reward[obs_idx]:
            reward = 1

        return self.stimuli[obs_idx], reward, done, self.info

    def reset(self):
        obs_idx = self.np_random.choice(range(len(self.stimuli)), 
                                   p=self.p_stimuli, replace=True)
        return self.stimuli[obs_idx]

    def render(self, mode='human', close=False):
        pass