import sciunit
from gym import spaces
from .continuous import Continuous


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
        value : None or int or gym.spaces.Discrete
            observation_space set to class gym.spaces.Discrete when passed None 
            (default). With a int or an instance of gym.spaces.Discrete, observation_space 
            will be set to the gym.spaces.Discrete accordingly.
        """
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
        """
        Returns
        -------
        int or None
            Models only understand discrete observation up to n_obs. Return observation_space.n when
            observation_space present, None otherwise.
        """
        if isinstance(self.observation_space, spaces.Discrete):
            return self.observation_space.n
        else:
            return None

    @n_obs.setter
    def n_obs(self, value):
        """
        Parameters
        ----------
        value : None or int
            Set the observation_space to gym.spaces.Discrete when passed None 
            (default). With a int, the model's observation_space will be set 
            accordingly.
        """
        if not value:
            self.observation_space = None
        elif not isinstance(value, int):
            raise TypeError('n_obs must be an integer')
        elif isinstance(value, int):
            self.observation_space = value

    def _check_observation(self, x):
        """
        Check whether given value is a valid observation.
        """
        if not isinstance(x, list) and all(isinstance(i, int) for i in x):
            raise AssertionError("Data must be list of integers.")
        return True

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
        """
        Returns
        -------
        int or None
            Models only understand discrete actions up to n_action.
            Return action_space.n when action_space present, None otherwise.
        """
        if isinstance(self.action_space, spaces.Discrete):
            return self.action_space.n
        else:
            return None

    @n_action.setter
    def n_action(self, value):
        """
        Parameters
        ----------
        value : int or None
            set the action_space to gym.spaces.Discrete when passed None 
            (default). With a int, the model's action_space will be set 
            accordingly.
        """
        if not value:
            self.action_space = None
        elif not isinstance(value, int):
            raise TypeError('n_action must be an integer')
        elif isinstance(value, int):
            self.action_space = value

    def _check_action(self, x):
        """
        Check whether given value is a valid action.
        """
        if not isinstance(x, list) and all(isinstance(i, int) for i in x):
            raise AssertionError("Data must be list of integers.")
        return True

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
        """
        Returns
        -------
        int or None
            Models only understand n_obs observation in a gym.spaces.MultiBinary space. 
            Return observation_space.n when observation_space present, None otherwise.
        """
        if isinstance(self.observation_space, spaces.MultiBinary):
            return self.observation_space.n
        else:
            return None

    @n_obs.setter
    def n_obs(self, value):
        """
        Parameters
        ----------
        value : int or None
            set the observation_space to gym.spaces.MultiBinary when passed None 
            (default). With a int, the model's observation_space will be set 
            accordingly.
        """
        if not value:
            self.observation_space = None
        elif not isinstance(value, int):
            raise TypeError('n_obs must be an integer')
        elif isinstance(value, int):
            self.observation_space = value

    def _check_observation(self, x):
        """
        Check whether given value is a valid obeservation.
        """
        if not isinstance(x, list) and all(spaces.MultiBinary(1).contains(i) for i in x):
            raise AssertionError("Data must be list of MultiBinary.")
        return True

class Continuous(ActionSpace):
    """
    Capability to understand continuous actions (i.e. R^1 in continuous).
    """

    @property
    def action_space(self):
        return Continuous()

    def _check_action(self, x):
        """
        Check whether given value is a valid action.
        """
        if not isinstance(x, list) and all(self.action_space.contains(i) for i in x):
            raise AssertionError("Data must be list of continuous.")
        return True

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
