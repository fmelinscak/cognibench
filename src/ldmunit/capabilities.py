import sciunit
import numpy as np
from gym import spaces
from .continuous import ContinuousSpace


class Interactive(sciunit.Capability):
    """Capability to interact with an environment (i.e. gym.Env)

    Models with this capability are required to have the following methods to be 
    able to respond to environment and update themselves in an interactive manner.
    """

    def predict(self, *args, **kwargs):
        """
        Given stimulus, model should return the prediction.
        """
        raise NotImplementedError("Must implement predict.")
    
    def reset(self, *args, **kwargs):
        """
        Reset the hidden state of the model to initial state.
        """
        raise NotImplementedError("Must implement reset.")

    def update(self, *args, **kwargs):
        """
        Given stimulus, rewards and action, the model should updates its 
        hidden state. Also named evolution function in some packages.
        """
        raise NotImplementedError("Must implement update.")

    def act(self, *args, **kwargs):
        """
        For decision making, return the action taken by the model.
        Associative learning models should return the predicted value.
        Also named observation function in some packages.
        """
        raise NotImplementedError("Must implement act")

class LogProbModel(sciunit.Capability):
    """
    Capability for models that produce a log probability distribution
    as a result of their predict function.

    Models with this capability are required to have the following methods
    """
    def predict(self, *args, **kwargs):
        """
        Given stimulus, model should return log pdf or log pmf function.
        """
        raise NotImplementedError("Must implement predict returning log-pdf or log-pmf.")

class ActionSpace(sciunit.Capability):
    """
    Capability to understand action in a given space (gym.spaces).
    """

    @property
    def action_space(self):
        """
        Returns
        -------
        gym.spaces
            Models only understand action in this space.
        """
        raise NotImplementedError("Must have action_space attribute.")

class ObservationSpace(sciunit.Capability):
    """
    Capability to understand action in a given space (gym.spaces).
    """

    @property
    def observation_space(self):
        """
        Returns
        -------
        gym.spaces
            Models only understand observation/stimulus in this space.
        """
        raise NotImplementedError("Must have observation_space attribute.")

class DiscreteObservation(ObservationSpace):
    """
    Capability to understand observation/stimulus/cues in a given discrete space (gym.spaces.Discrete).
    """

    @property
    def observation_space(self):
        """
        Returns
        -------
        gym.spaces.Discrete
            Models only understand observation in the gym.spaces.Discrete set.
        """
        return self._observation_space

    @observation_space.setter
    def observation_space(self, value):
        """
        Set the gym.spaces.Discrete set.

        Parameters
        ----------
        value : int or gym.spaces.Discrete
            Observation_space will be set to the gym.spaces.Discrete accordingly.
        """
        if isinstance(value, spaces.Discrete):
            self._observation_space = value
        elif np.issubdtype(type(value), np.integer):
            self._observation_space = spaces.Discrete(value)
        else:
            raise TypeError("observation_space must be integer or gym.spaces.Discrete")

    @property
    def n_obs(self):
        """
        Returns
        -------
        int
            Dimension of the observation space.
        """
        return self.observation_space.n

    def _check_observation(self, values):
        """
        Check whether given values are valid observations.
        """
        return all(np.issubdtype(type(x), np.integer) for x in values)

class DiscreteAction(ActionSpace):
    """
    Capability to understand action in a given discrete space (gym.spaces.Discrete).
    """

    @property
    def action_space(self):
        """
        Returns
        -------
        gym.spaces.Discrete
            Models only understand action in the gym.spaces.Discrete set.
        """
        return self._action_space

    @action_space.setter
    def action_space(self, value):
        """
        Parameters
        ----------
        value : None, int, gym.spaces.Discrete
            observation_space set to class gym.spaces.Discrete when passed None 
            (default). With a int or an instance of gym.spaces.Discrete, observation_space 
            will be set to the gym.spaces.Discrete accordingly.
        """
        if isinstance(value, spaces.Discrete):
            self._action_space = value
        elif np.issubdtype(type(value), np.integer):
            self._action_space = spaces.Discrete(value)
        else:
            raise TypeError("action_space must be integer or gym.spaces.Discrete")

    @property
    def n_action(self):
        """
        Returns
        -------
        int
            Dimension of the action space.
        """
        return self.action_space.n

    def _check_action(self, values):
        """
        Check whether given value is a valid action.
        """
        return all(np.issubdtype(type(x), np.integer) for x in values)

class MultiBinaryObservation(ObservationSpace):
    """
    Capability to understand actions in a given multi-binary space (gym.spaces.MultiBinary).
    """

    @property
    def observation_space(self):
        """
        Returns
        -------
        gym.spaces.MultiBinary
            Models only understand observation/stimulus in the gym.spaces.MultiBinary set.
            For example np.array([0, 1, 0, 1])

        Example
        -------
        >>> import ldmunit.capabilities as cap
        >>> obs = cap.MultiBinaryObservation()
        >>> obs.observation_space = 4
        >>> obs.observation_space
        MultiBinary(4)
        """
        return self._observation_space

    @observation_space.setter
    def observation_space(self, value):
        """
        Parameters
        ----------
        value : None or int or gym.spaces.MultiBinary
            observation_space set to class gym.spaces.MultiBinary when passed None 
            (default). With a int or an instance of gym.spaces.MultiBinary, observation_space 
            will be set to the gym.spaces.MultiBinary accordingly.
        """
        if isinstance(value, spaces.MultiBinary):
            self._observation_space = value
        elif isinstance(value, int):
            self._observation_space = spaces.MultiBinary(value)
        else:
            raise TypeError("action_space must be integer or gym.spaces.MultiBinary")

    @property
    def n_obs(self):
        """
        Returns
        -------
        int
            Dimension of the observation space.
        """
        return self.observation_space.n

    def _check_observation(self, values):
        """
        Check whether given value is a valid obeservation.
        """
        return all(self.observation_space.contains(x) for x in values)

class ContinuousAction(ActionSpace):
    """
    Capability to understand continuous actions (i.e. R^1 in continuous).
    """

    @property
    def action_space(self):
        return ContinuousSpace()

    def _check_action(self, values):
        """
        Check whether given value is a valid action.
        """
        return all(self.action_space.contains(x) for x in values)
