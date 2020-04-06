import matlab.engine
import time
import matlab
import numpy as np

# This is actually `matlab._internal`, but matlab/__init__.py
# mangles the path making it appear as `_internal`.
# Importing it under a different name would be a bad idea.
from _internal.mlarray_utils import _get_strides, _get_mlsize

from functools import partial
from cognibench.logging import logger


_matlab_sess = None


class MatlabWrapperMixin:
    """
    Mixin class that allows easy porting of Matlab models to CogniBench model interface.

    It is assumed that all model methods are implemented as separate functions under some directory. For example,
    one can create a model consisting reset, predict, fit, update and act functions using the below directory structure

        model/
            -- reset.m
            -- predict.m
            -- fit.m
            -- update.m
            -- act.m

    and mapping each function to its corresponding filename in class `__init__` method.
    """

    def __init__(
        self,
        *args,
        lib_paths,
        import_base_path,
        reset_fn=None,
        predict_fn=None,
        fit_fn=None,
        update_fn=None,
        act_fn=None,
        **kwargs,
    ):
        """
        Initialize the Matlab wrapper to register function mappings and define the class interface. Each initialized
        object deriving from this wrapper class will have its own Matlab process.

        Parameters
        ----------
        lib_paths : list of str
            Folders containing additional libraries required by the model. Each path will be called with addpath.

        import_base_path : str
            Base folder path containing Matlab function implementations.

        reset_fn : str or callable
            If string, then a function of the same name that exists in the given import path will be registered
            to the `reset` method of the class. If a function, calls to `reset` are forwarded to this function with the
            additional Matlab session as the first argument. For example, if normally `reset` is called as `self.reset(a, b, c)`,
            then this function will receive `reset_fn(matlab_session, a, b, c)`.

        predict_fn : str or callable
            Analogous to reset_fn documentation.

        fit_fn : str or callable
            Analogous to reset_fn documentation.

        update_fn : str or callable
            Analogous to reset_fn documentation.

        act_fn : str or callable
            Analogous to reset_fn documentation.
        """
        global _matlab_sess
        if _matlab_sess is None:
            logger().info("Initializing MATLAB session.")
            _matlab_sess = matlab.engine.start_matlab()
        for path in lib_paths:
            _matlab_sess.addpath(path, nargout=0)
        _matlab_sess.addpath(import_base_path, nargout=0)

        _define_if_given(self, reset_fn, "reset")
        _define_if_given(self, predict_fn, "predict")
        _define_if_given(self, fit_fn, "fit")
        _define_if_given(self, update_fn, "update")
        _define_if_given(self, act_fn, "act")

        super().__init__(*args, **kwargs)


def _define_if_given(obj, fn, fn_name_to_set):
    """
    Define `obj.fn_name_to_set` if fn is not None.
    """
    if fn is None:
        return
    if isinstance(fn, str):
        fn_to_call = partial(_matlab_sess.feval, fn)
    else:

        def fn_to_call(*args, **kwargs):
            return fn(_matlab_sess, *args, **kwargs)

    setattr(obj, fn_name_to_set, _matlab_auto_convert(fn_to_call))


def _apply_recursively(func, structure):
    if isinstance(structure, list):
        return [_apply_recursively(func, x) for x in structure]
    if isinstance(structure, tuple):
        return tuple(_apply_recursively(func, x) for x in structure)
    if isinstance(structure, dict):
        return {k: _apply_recursively(func, v) for k, v in structure.items()}
    return func(structure)


def _as_matlab_if_possible(x):
    try:
        return as_matlab(x)
    except (TypeError, AttributeError):
        return x


def _matlab_auto_convert(func):
    def ans(*args, **kwargs):
        m_args = _apply_recursively(_as_matlab_if_possible, args)
        m_kwargs = _apply_recursively(_as_matlab_if_possible, kwargs)
        m_out = func(*m_args, **m_kwargs)
        return _apply_recursively(np.asarray, m_out)

    return ans


# NUMPY TO MATLAB CONVERSION
# --------------------------
# Implementation copied from the excellent answer at
# https://stackoverflow.com/questions/10997254/converting-numpy-arrays-to-matlab-and-vice-versa
def _wrapper__init__(self, arr):
    assert arr.dtype == type(self)._numpy_type
    self._python_type = type(arr.dtype.type().item())
    self._is_complex = np.issubdtype(arr.dtype, np.complexfloating)
    self._size = _get_mlsize(arr.shape)
    self._strides = _get_strides(self._size)[:-1]
    self._start = 0

    if self._is_complex:
        self._real = arr.real.ravel(order="F")
        self._imag = arr.imag.ravel(order="F")
    else:
        self._data = arr.ravel(order="F")


_wrappers = {}


def _define_wrapper(matlab_type, numpy_type):
    t = type(
        matlab_type.__name__,
        (matlab_type,),
        dict(__init__=_wrapper__init__, _numpy_type=numpy_type),
    )
    # this tricks matlab into accepting our new type
    t.__module__ = matlab_type.__module__
    _wrappers[numpy_type] = t


_define_wrapper(matlab.double, np.double)
_define_wrapper(matlab.single, np.single)
_define_wrapper(matlab.uint8, np.uint8)
_define_wrapper(matlab.int8, np.int8)
_define_wrapper(matlab.uint16, np.uint16)
_define_wrapper(matlab.int16, np.int16)
_define_wrapper(matlab.uint32, np.uint32)
_define_wrapper(matlab.int32, np.int32)
_define_wrapper(matlab.uint64, np.uint64)
_define_wrapper(matlab.int64, np.int64)
_define_wrapper(matlab.logical, np.bool_)


def as_matlab(arr):
    try:
        cls = _wrappers[arr.dtype.type]
    except KeyError:
        raise TypeError("Unsupported data type")
    return cls(arr)
