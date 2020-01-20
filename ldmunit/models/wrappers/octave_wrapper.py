from oct2py import Oct2Py
from functools import partial


class OctaveWrapperMixin:
    """
    Mixin class that allows easy porting of Octave models to LDMUnit model interface.

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
        import_base_path,
        reset_fn=None,
        predict_fn=None,
        fit_fn=None,
        update_fn=None,
        act_fn=None,
        **kwargs,
    ):
        """
        Initialize the Octave wrapper to register function mappings and define the class interface. Each initialized
        object deriving from this wrapper class will have its own Octave process.

        Parameters
        ----------
        import_base_path : str
            Base folder path containing Octave function implementations.

        reset_fn : str or callable
            If string, then a function of the same name that exists in the given import path will be registered
            to the `reset` method of the class. If a function, calls to `reset` are forwarded to this function with the
            additional Octave session as the first argument. For example, if normally `reset` is called as `self.reset(a, b, c)`,
            then this function will receive `reset_fn(octave_session, a, b, c)`.

        predict_fn : str or callable
            Analogous to reset_fn documentation.

        fit_fn : str or callable
            Analogous to reset_fn documentation.

        update_fn : str or callable
            Analogous to reset_fn documentation.

        act_fn : str or callable
            Analogous to reset_fn documentation.
        """
        self.octave_session = Oct2Py()
        self.octave_session.eval(f'addpath("{import_base_path}");')

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
        fn_to_call = partial(obj.octave_session.feval, fn)
    else:

        def fn_to_call(*args, **kwargs):
            return fn(obj.octave_session, *args, **kwargs)

    setattr(obj, fn_name_to_set, fn_to_call)
