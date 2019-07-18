import functools
import numpy as np


def partialclass(cls, *args, **kwargs):
    """
    Partially initialize a class by binding its `__init__` method with the given input arguments. The returned class
    can be initialized by passing the remaining arguments required for initialization.

    Parameters
    ----------
    cls : type
        Any class type.

    Returns
    -------
    class
        A new class that is partially initialized using the `*args` and `**kwargs` arguments.

    See Also
    --------
    `<https://stackoverflow.com/questions/38911146/python-equivalent-of-functools-partial-for-a-class-constructor>`_.
    """
    class OutCls(cls):
        __init__ = functools.partialmethod(cls.__init__, *args, **kwargs)

    return OutCls


def is_arraylike(x):
    """
    Returns
    -------
    bool
        True if the input is an array-like object.

    See Also
    --------
    `<https://docs.scipy.org/doc/numpy/reference/generated/numpy.isscalar.html>`_.
    """
    return np.ndim(x) != 0