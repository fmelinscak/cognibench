import sciunit
import numpy as np
from gym import spaces
from .continuous import ContinuousSpace
from overrides import overrides


class Interactive(sciunit.Capability):
    """Capability to interact with an environment

    Models with this capability are required to have the following methods to be
    able to respond to environment and update themselves in an interactive manner.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def update(self, *args, **kwargs):
        """
        Given stimulus, rewards and action, the model should update its
        hidden state. Also named evolution function in some packages.
        """
        raise NotImplementedError("Must implement update.")


class PredictsLogpdf(sciunit.Capability):
    """
    Capability for models that produce a logpdf as the return value of their predict method.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    pass


class MultiSubjectModel(sciunit.Capability):
    """
    Capability for models that natively support multi subject data. These types of models
    must take a subject index as the first argument to their core functions such as fit, predict, reset, etc. The
    names of these multi-subject functions must be stored in multi_subject_methods variable as given below.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    multi_subject_methods = []

    def fit_jointly(self, *args, **kwargs):
        """
        Take the data of all the subjects at once and fit the multi-subject model, either by fitting each model
        separately, or fitting all the models at once.
        """
        raise NotImplementedError("Multi-subject model must implement fit_multisubject")


class ReturnsNumParams(sciunit.Capability):
    """
    Capability for models that can return the number of parameters they utilize.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def n_params(self):
        """
        The model should return the number of parameters it has.
        """
        raise NotImplementedError("Must have n_params attribute.")


class ActionSpace(sciunit.Capability):
    """
    Capability to understand action in a given space (:class:`gym.spaces.Space`).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_action_space(self):
        """
        Returns
        -------
        :class:`gym.spaces.Space`
            Models only understand action in this space.
        """
        raise NotImplementedError("Must have action_space attribute.")

    def set_action_space(self, x):
        raise NotImplementedError("Must have action_space attribute.")


class ObservationSpace(sciunit.Capability):
    """
    Capability to understand action in a given space (:class:`gym.spaces.Space`).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_observation_space(self):
        """
        Returns
        -------
        :class:`gym.spaces.Space`
            Models only understand observation/stimulus in this space.
        """
        raise NotImplementedError("Must have observation_space attribute.")

    def set_observation_space(self, x):
        raise NotImplementedError("Must have observation_space attribute.")


class DiscreteObservation(ObservationSpace):
    """
    Capability to understand observation/stimulus/cues in a given discrete space (:class:`gym.spaces.Discrete`).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @overrides
    def get_observation_space(self):
        """
        Returns
        -------
        :class:`gym.spaces.Discrete`
            Models only understand observation in the :class:`gym.spaces.Discrete` set.
        """
        return self._observation_space

    @overrides
    def set_observation_space(self, value):
        """
        Set the :class:`gym.spaces.Discrete` set.

        Parameters
        ----------
        value : int or :class:`gym.spaces.Discrete`
            Observation_space will be set to the :class:`gym.spaces.Discrete` accordingly.
        """
        if isinstance(value, spaces.Discrete):
            self._observation_space = value
        elif np.issubdtype(type(value), np.integer):
            assert value > 0, "observation_space must be positive"
            self._observation_space = spaces.Discrete(value)
        else:
            raise TypeError("observation_space must be integer or gym.spaces.Discrete")

    def n_obs(self):
        """
        Returns
        -------
        int
            Dimension of the observation space.
        """
        return self.get_observation_space().n

    def _check_observation(self, values):
        """
        Check whether given values are valid observations.
        """
        return all(np.issubdtype(type(x), np.integer) for x in values)


class DiscreteAction(ActionSpace):
    """
    Capability to understand action in a given discrete space (:class:`gym.spaces.Discrete`).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @overrides
    def get_action_space(self):
        """
        Returns
        -------
        :class:`gym.spaces.Discrete`
            Models only understand action in the :class:`gym.spaces.Discrete` set.
        """
        return self._action_space

    @overrides
    def set_action_space(self, value):
        """
        Parameters
        ----------
        value : None, int, :class:`gym.spaces.Discrete`
            observation_space set to class :class:`gym.spaces.Discrete` when passed None
            (default). With a int or an instance of :class:`gym.spaces.Discrete`, observation_space
            will be set to the :class:`gym.spaces.Discrete` accordingly.
        """
        if isinstance(value, spaces.Discrete):
            self._action_space = value
        elif np.issubdtype(type(value), np.integer):
            assert value > 0, "action_space must be positive"
            self._action_space = spaces.Discrete(value)
        else:
            raise TypeError("action_space must be integer or gym.spaces.Discrete")

    def n_action(self):
        """
        Returns
        -------
        int
            Dimension of the action space.
        """
        return self.get_action_space().n

    def _check_action(self, values):
        """
        Check whether given value is a valid action.
        """
        return all(np.issubdtype(type(x), np.integer) for x in values)


class MultiBinaryObservation(ObservationSpace):
    """
    Capability to understand actions in a given multi-binary space (:class:`gym.spaces.MultiBinary`).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @overrides
    def get_observation_space(self):
        """
        Returns
        -------
        :class:`gym.spaces.MultiBinary`
            Models only understand observation/stimulus in the :class:`gym.spaces.MultiBinary` set.
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

    @overrides
    def set_observation_space(self, value):
        """
        Parameters
        ----------
        value : None or int or :class:`gym.spaces.MultiBinary`
            observation_space set to class :class:`gym.spaces.MultiBinary` when passed None
            (default). With a int or an instance of :class:`gym.spaces.MultiBinary`, observation_space
            will be set to the :class:`gym.spaces.MultiBinary` accordingly.
        """
        if isinstance(value, spaces.MultiBinary):
            self._observation_space = value
        elif isinstance(value, int):
            assert value > 0, "observation_space must be positive"
            self._observation_space = spaces.MultiBinary(value)
        else:
            raise TypeError("action_space must be integer or gym.spaces.MultiBinary")

    def n_obs(self):
        """
        Returns
        -------
        int
            Dimension of the observation space.
        """
        return self.get_observation_space().n

    def _check_observation(self, values):
        """
        Check whether given value is a valid obeservation.
        """
        return all(self.get_observation_space().contains(x) for x in values)


class ContinuousAction(ActionSpace):
    """
    Capability to understand continuous actions (i.e. :math:`\mathbb{R}` in continuous).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @overrides
    def get_action_space(self):
        return self._action_space

    @overrides
    def set_action_space(self, value):
        if isinstance(value, ContinuousSpace):
            self._action_space = value
        elif isinstance(value, tuple):
            self._action_space = ContinuousSpace(shape=value)
        else:
            raise TypeError(
                "observation_space must be a shape tuple or ContinuousSpace"
            )

    def _check_action(self, values):
        """
        Check whether given value is a valid action.
        """
        return all(self.get_action_space().contains(x) for x in values)


class ContinuousObservation(ObservationSpace):
    """
    Capability to understand continuous observations (i.e. :math:`\mathbb{R}` in continuous).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @overrides
    def get_observation_space(self):
        return self._observation_space

    @overrides
    def set_observation_space(self, value):
        if isinstance(value, ContinuousSpace):
            self._observation_space = value
        elif isinstance(value, tuple):
            self._observation_space = ContinuousSpace(shape=value)
        else:
            raise TypeError(
                "observation_space must be a shape tuple or ContinuousSpace"
            )

    def _check_observation(self, values):
        """
        Check whether given value is a valid observation.
        """
        return all(self.get_observation_space().contains(x) for x in values)
