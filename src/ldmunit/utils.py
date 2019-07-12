import functools


def partialclass(cls, *args, **kwargs):
    """
    Partially initialize the given class cls using input arguments *args
    and **kwargs. Return the resulting class.

    https://stackoverflow.com/questions/38911146/python-equivalent-of-functools-partial-for-a-class-constructor
    """
    class OutCls(cls):
        __init__ = functools.partialmethod(cls.__init__, *args, **kwargs)

    return OutCls
