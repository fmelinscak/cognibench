import sciunit
from gym import spaces
from .continous import Continous

### The Capabilities ###


class Interactive(sciunit.Capability):
    """
    Description:
        Requires models to respound to the environment in a interactive manner.

    For decision making cases, the model is init with the number of actions and
    number of observation.
    For associate learning models, the model is initialized with number of 
    observation.


    Observation:
        Cue/stimulus in other setting or literature. 
        Type: defined in each model.
        
    Actions:
        Type: defined in each model.
        
    Reward:
        In the decision making model setting, reward is 1 when correct action is chosen.
    """

    def predict(self):
        """Given stimulus, model should return the function for the calculation
        of log probability density function or log probability mass function.
        """
        raise NotImplementedError("Must implement predict.")
    
    def reset(self):
        """Reset the hidden state of the model to initial state.
        """
        raise NotImplementedError("Must implement reset.")

    def update(self):
        """Given stimulus, rewards and action, the model should updates its 
        hidden state. Also named evolution function in some packages.
        """
        raise NotImplementedError("Must implement update.")

    def act(self):
        """For decision making, return the action taken by the model,
        Associative learning models returns the predicted value.
        Also named observation function in some packages.
        """
        raise NotImplementedError("Must implement act")

class ActionSpace(sciunit.Capability):
    @property
    def action_space(self):
        raise NotImplementedError("Must have action_space attribute.")

class ObservationSpace(sciunit.Capability):
    @property
    def observation_space(self):
        raise NotImplementedError("Must have observation_space attribute.")

class DiscreteObservation(ObservationSpace):

    @property
    def observation_space(self):
        return self._observation_space

    @observation_space.setter
    def observation_space(self, value):
        if not value:
            self._observation_space = spaces.Discrete
        elif isinstance(value, spaces.Discrete):
            self._observation_space = value
        elif isinstance(value, int):
            self._observation_space = spaces.Discrete(value)
        else:
            raise TypeError("action_space must be integer or gym.spaces.Discrete")

    @observation_space.deleter
    def observation_space(self):
        del self._observation_space

    @property
    def n_obs(self):
        if isinstance(self.observation_space, spaces.Discrete):
            return self.observation_space.n
        else:
            return None

    @n_obs.setter
    def n_obs(self, value):
        if not value:
            self.observation_space = None
        elif not isinstance(value, int):
            raise TypeError('n_obs must be an integer')
        elif isinstance(value, int):
            self.observation_space = value

    def _check_observation(self, x):
        if not isinstance(x, list) and all(isinstance(i, int) for i in x):
            raise AssertionError("Data must be list of integers.")
        return True

class DiscreteAction(ActionSpace):

    @property
    def action_space(self):
        return self._action_space

    @action_space.setter
    def action_space(self, value):
        if not value:
            self._action_space = spaces.Discrete
        elif isinstance(value, spaces.Discrete):
            self._action_space = value
        elif isinstance(value, int):
            self._action_space = spaces.Discrete(value)
        else:
            raise TypeError("action_space must be integer or gym.spaces.Discrete")

    @action_space.deleter
    def action_space(self):
        del self._action_space

    @property
    def n_action(self):
        if isinstance(self.action_space, spaces.Discrete):
            return self.action_space.n
        else:
            return None

    @n_action.setter
    def n_action(self, value):
        if not value:
            self.action_space = None
        elif not isinstance(value, int):
            raise TypeError('n_action must be an integer')
        elif isinstance(value, int):
            self.action_space = value

    def _check_action(self, x):
        if not isinstance(x, list) and all(isinstance(i, int) for i in x):
            raise AssertionError("Data must be list of integers.")
        return True

class MultiBinaryObservation(ObservationSpace):

    @property
    def observation_space(self):
        return self._observation_space

    @observation_space.setter
    def observation_space(self, value):
        if not value:
            self._observation_space = spaces.MultiBinary
        elif isinstance(value, spaces.MultiBinary):
            self._observation_space = value
        elif isinstance(value, int):
            self._observation_space = spaces.MultiBinary(value)
        else:
            raise TypeError("action_space must be integer or gym.spaces.MultiBinary")

    @observation_space.deleter
    def observation_space(self):
        del self._observation_space

    @property
    def n_obs(self):
        if isinstance(self.observation_space, spaces.MultiBinary):
            return self.observation_space.n
        else:
            return None

    @n_obs.setter
    def n_obs(self, value):
        if not value:
            self.observation_space = None
        elif not isinstance(value, int):
            raise TypeError('n_obs must be an integer')
        elif isinstance(value, int):
            self.observation_space = value

    def _check_observation(self, x):
        if not isinstance(x, list) and all(spaces.MultiBinary(1).contains(i) for i in x):
            raise AssertionError("Data must be list of MultiBinary.")
        return True

class ContinousAction(ActionSpace):

    @property
    def action_space(self):
        return Continous() # self._action_space

    def _check_action(self, x):
        if not isinstance(x, list) and all(self.action_space.contains(i) for i in x):
            raise AssertionError("Data must be list of continous.")
        return True