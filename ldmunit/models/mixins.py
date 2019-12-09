import numpy as np
from scipy.optimize import minimize
from ldmunit.utils import negloglike, is_arraylike
from ldmunit.logging import logger
from ldmunit.capabilities import ReturnsNumParams


class ParametricModelMixin(ReturnsNumParams):
    """
    A simple mixin class that allows easy ReturnsNumParams interface implementation
    for parametric models. It is assumed that the deriving class has a sequence or dictionary
    field `self.paras` which stores all the parameters of the model separately. For more
    sophisticated models, implementing the ReturnsNumParams interface yourself may be easier and
    more accurate.
    """

    def n_params(self):
        return len(self.paras)


class ReinforcementLearningFittingMixin:
    """
    Mixin class to automatically add negative log likelihood model fitting functionality to interactive models.

    See Also
    --------
    :func:`scipy.optimize.minimize`
    """

    def fit(self, stimuli, rewards, actions, method="Nelder-Mead", **kwargs):
        # TODO: this relies on PredictsLogPdf and Interactive capabilities. How to specify that?
        # TODO: how to specify other scores instead of negloglike?
        # TODO: Maybe mixin should be capable of using various scores to optimize?
        def f(x, lens):
            _flatten_array_into_dict(self.paras, x, lens)
            predictions = []
            # TODO: essentially the same logic as InteractiveTesting; refactor?
            self.reset()
            for s, r, a in zip(stimuli, rewards, actions):
                predictions.append(self.predict(s))
                self.update(s, r, a)
            return negloglike(actions, predictions)

        x0, lens = _flatten_dict_into_array(self.paras)
        # TODO: make this modifiable from outside
        options = {"maxiter": 1}
        opt_res = minimize(
            f, x0, args=(lens,), method=method, options=options, **kwargs
        )
        if not opt_res.success:
            logger().debug(
                f"{method} fitting on {self.name} has not finished successfully! Cause of termination: {opt_res.message}"
            )

        _flatten_array_into_dict(self.paras, opt_res.x, lens)

        logger().debug(
            f"Model parameters has been set to the outputs of optimization procedure."
        )


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
