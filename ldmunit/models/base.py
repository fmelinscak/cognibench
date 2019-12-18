from ldmunit.logging import logger
from ldmunit.utils import negloglike, is_arraylike
from scipy.optimize import minimize
import sciunit
import numpy as np
from gym.utils import seeding
from collections.abc import Mapping
from gym import spaces
from ldmunit.capabilities import (
    Interactive,
    ProducesPolicy,
    PredictsLogpdf,
    ReturnsNumParams,
)
from ldmunit.continuous import ContinuousSpace
from overrides import overrides


class LDMModel(sciunit.Model):
    """
    Helper base class for LDMUnit models.
    """

    @overrides
    def __init__(self, seed=None, **kwargs):
        """
        Parameters
        ----------
        seed : int
            Random seed. Must be a nonnegative integer. If seed is None,
            random state is set randomly by gym.utils.seeding. (Default: None)
        """
        self.seed = seed
        super().__init__(**kwargs)

    @property
    def seed(self):
        """
        Returns
        -------
        int or None
            Random seed used to initialize the random number generator.
            Seed is None only if it was omitted during model initialization.
        """
        return self._seed

    @property
    def rng(self):
        """
        Returns
        -------
        :class:`numpy.random.RandomState`
            Random number generator state. Use this object as an np.random
            replacement to generate random numbers. This way, you can reproduce
            your results if you always use the same seed during model initialization.
        """
        return self._rng

    @seed.setter
    def seed(self, value):
        self._seed = value
        self._rng, _ = seeding.np_random(seed=value)

    def fit(self, *args, **kwargs):
        """
        Fit the model to a batch of stimuli. If this is a multi-subject model, then the stimuli should be a list
        where each element contains the stimuli of the corresponding subject.

        By default, this method does not perform model fitting. This method should be overridden only if you need
        model fitting functionality.
        """
        pass

    def predict(self, *args, **kwargs):
        """
        Make a prediction given a stimulus.
        """
        raise NotImplementedError("Must implement predict.")

    def reset(self):
        """
        Reset the hidden state of the model. Subclasses should override
        this method with suitable default hidden state values so that hidden
        state is set to this default during object initialization.
        """
        pass


class LDMAgent:
    def __init__(self, *args, paras_dict=None, seed=None, **kwargs):
        self.seed = seed
        self.paras = paras_dict
        super().__init__(*args, **kwargs)

    @property
    def seed(self):
        """
        Returns
        -------
        int or None
            Random seed used to initialize the random number generator.
            Seed is None only if it was omitted during model initialization.
        """
        return self._seed

    @property
    def rng(self):
        """
        Returns
        -------
        :class:`numpy.random.RandomState`
            Random number generator state. Use this object as an np.random
            replacement to generate random numbers. This way, you can reproduce
            your results if you always use the same seed during model initialization.
        """
        return self._rng

    @seed.setter
    def seed(self, value):
        self._seed = value
        self._rng, _ = seeding.np_random(seed=value)

    def act(self, *args, **kwargs):
        raise NotImplementedError("LDMAgent must implement act")

    def update(self, *args, **kwargs):
        raise NotImplementedError("LDMAgent must implement update")

    def reset(self):
        self.hidden_state = dict()

    @property
    def paras(self):
        return self._paras

    @paras.setter
    def paras(self, paras_dict):
        self._paras = paras_dict

    @property
    def hidden_state(self):
        return self._hidden_state

    @hidden_state.setter
    def hidden_state(self, state):
        self._hidden_state = state


class PolicyBasedModel(LDMModel, Interactive, PredictsLogpdf, ReturnsNumParams):
    # TODO: adapt action and obs spaces according to the Agent
    def __init__(self, *args, agent, param_initializer, **kwargs):
        assert isinstance(
            agent, ProducesPolicy
        ), "PolicyBasedModel can only accept agents satisfying ProducesPolicy capability"
        super().__init__(*args, **kwargs)
        self.agent = agent
        self.param_initializer = param_initializer

    @overrides
    def n_params(self):
        return len(self.agent.paras)

    @overrides
    def reset(self):
        self.agent.reset()

    @overrides
    def fit(self, stimuli, rewards, actions):
        if isinstance(self.param_initializer, Mapping):
            paras_init = self.param_initializer
        else:
            paras_init = self.param_initializer(seed=self.seed)

        self.agent.paras = paras_init

        def f(x, lens):
            _flatten_array_into_dict(self.agent.paras, x, lens)
            predictions = []
            # TODO: essentially the same logic as InteractiveTesting; refactor?
            self.reset()
            for s, r, a in zip(stimuli, rewards, actions):
                predictions.append(self.predict(s))
                self.update(s, r, a)
            return negloglike(actions, predictions)

        x0, lens = _flatten_dict_into_array(self.agent.paras)
        # TODO: make this modifiable from outside
        opt_res = minimize(
            f, x0, args=(lens,), method="Nelder-Mead", options={"maxiter": 2}
        )
        if not opt_res.success:
            logger().debug(
                f"Fitting on {self.name} has not finished successfully! Cause of termination: {opt_res.message}"
            )

        _flatten_array_into_dict(self.agent.paras, opt_res.x, lens)

        logger().debug(
            f"Agent parameters has been set to the outputs of optimization procedure."
        )

    @overrides
    def predict(self, stimulus):
        policy = self.agent.eval_policy(stimulus)
        return policy.logpdf if hasattr(policy, "logpdf") else policy.logpmf

    def update(self, stimulus, reward, action, done=False):
        return self.agent.update(stimulus, reward, action, done)


def _flatten_array_into_dict(dictionary, arr, lens):
    i = 0
    for k, currlen in zip(dictionary.keys(), lens):
        if currlen == 1:
            dictionary[k] = arr[i]
        else:
            dictionary[k] = arr[i : i + currlen]
        i += currlen


def _flatten_dict_into_array(dictionary, dtype=np.float32):
    lens = np.array(
        [len(v) if is_arraylike(v) else 1 for v in dictionary.values()], dtype=np.int32
    )
    arr = np.empty(np.sum(lens), dtype=dtype)
    i = 0
    for v, currlen in zip(dictionary.values(), lens):
        arr[i : i + currlen] = v
        i += currlen
    return arr, lens
