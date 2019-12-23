from ldmunit.logging import logger
from ldmunit.utils import negloglike, is_arraylike
from scipy.optimize import minimize
import numpy as np
from collections.abc import Mapping
from gym import spaces
from ldmunit.capabilities import (
    Interactive,
    ProducesPolicy,
    PredictsLogpdf,
    ReturnsNumParams,
)
from ldmunit.continuous import ContinuousSpace
from ldmunit.models import LDMModel
from overrides import overrides


class PolicyModel(LDMModel, Interactive, PredictsLogpdf, ReturnsNumParams):
    # TODO: adapt action and obs spaces according to the Agent
    def __init__(self, *args, agent, **kwargs):
        assert isinstance(
            agent, ProducesPolicy
        ), "PolicyModel can only accept agents satisfying ProducesPolicy capability"
        super().__init__(*args, **kwargs)
        self.agent = agent
        self.init_paras()
        self.agent.reset()

    @overrides
    def n_params(self):
        return len(self.agent.paras)

    @overrides
    def reset(self):
        self.agent.reset()

    @overrides
    def set_paras(self, paras_dict):
        self.agent.paras = paras_dict

    @overrides
    def get_paras(self):
        return self.agent.paras

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

    @overrides
    def act(self, stimulus):
        return self.agent.act(stimulus)


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
