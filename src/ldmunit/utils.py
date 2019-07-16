import functools


def partialclass(cls, *args, **kwargs):
    """
    Partially initialize a by partially binding its __init__ method with the given input arguments. The returned class
    can be initialized by passing the remaining arguments required for initialization.

    Parameters
    ----------
    cls : type
        Any class type such as scores.SmallerBetterScore

    Returns
    -------
    class
        A new class that is partially initialized using the arguments
        passed in *args and **kwargs .

    See Also
    --------
    https://stackoverflow.com/questions/38911146/python-equivalent-of-functools-partial-for-a-class-constructor
    """
    class OutCls(cls):
        __init__ = functools.partialmethod(cls.__init__, *args, **kwargs)

    return OutCls