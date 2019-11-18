from oct2py import Oct2Py
from functools import partial


def _define_if_given(obj, fn, fn_name_to_set):
    if fn is None:
        return
    if isinstance(fn, str):
        fn_to_call = partial(obj.octave_session.feval, fn)
    else:

        def fn_to_call(*args, **kwargs):
            return fn(obj.octave_session, *args, **kwargs)

    setattr(obj, fn_name_to_set, fn_to_call)


class OctaveWrapperMixin:
    def __init__(
        self,
        *args,
        import_base_path,
        reset_fn=None,
        predict_fn=None,
        fit_fn=None,
        update_fn=None,
        act_fn=None,
        **kwargs,
    ):
        """
        If fn is str, a function of the same name that exists in the given import path will be called.
        If fn is a function, then it will be called with the created Octave session as the first argument, and the
        arguments of the corresponding function as the rest of the arguments.
        """
        self.octave_session = Oct2Py()
        self.octave_session.eval(f'addpath("{import_base_path}");')

        _define_if_given(self, reset_fn, "reset")
        _define_if_given(self, predict_fn, "predict")
        _define_if_given(self, fit_fn, "fit")
        _define_if_given(self, update_fn, "update")
        _define_if_given(self, act_fn, "act")
