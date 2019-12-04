from os import listdir
from os.path import join as pathjoin
from rpy2.robjects.packages import SignatureTranslatedAnonymousPackage as STAP
from rpy2.robjects import numpy2ri, pandas2ri, globalenv
from rpy2.robjects.environments import Environment as REnv


class RWrapperMixin:
    """
    Mixin class that allows easy porting of R models to LDMUnit model interface.
    It is assumed that all core model methods are implemented as separate functions under
    some directory.
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
        **kwargs
    ):
        """
        Initialize the R wrapper to register function mappings and define the class interface. All objects
        deriving from this interface share the same R process. However, from a user perspective, it is as if all
        the objects have their separate R processes since this object automatically isolates the environment of each
        object.

        Parameters
        ----------
        import_base_path : str
            Base folder path containing R function implementations.

        reset_fn : str or callable
            If string, then a function of the same name that exists in the given import path will be registered
            to the `reset` method of the class (i.e. for `reset.R`, just pass "reset"). If a function,
            calls to `reset` are forwarded to this function with the
            additional R module object as the first argument. For example, if normally `reset` is called as `self.reset(a, b, c)`,
            then this function will receive `reset_fn(R_module, a, b, c)`.

        predict_fn : str or callable
            Analogous to reset_fn documentation.

        fit_fn : str or callable
            Analogous to reset_fn documentation.

        update_fn : str or callable
            Analogous to reset_fn documentation.

        act_fn : str or callable
            Analogous to reset_fn documentation.
        """
        self.globalenv = REnv()
        _activate_conversions()

        # source all R files in the given directory, and create an R module from the merged sourcecode
        r_files = (f for f in listdir(import_base_path) if f.lower().endswith(".r"))
        r_codestring = ""
        for filename in r_files:
            with open(pathjoin(import_base_path, filename), "r") as f:
                r_codestring += f.read()
                r_codestring += "\n"
        self.R_module = STAP(r_codestring, "r_module")

        _define_if_given(self, reset_fn, "reset")
        _define_if_given(self, predict_fn, "predict")
        _define_if_given(self, fit_fn, "fit")
        _define_if_given(self, update_fn, "update")
        _define_if_given(self, act_fn, "act")


def _activate_conversions():
    numpy2ri.activate()
    pandas2ri.activate()


def _deactivate_conversions():
    numpy2ri.deactivate()
    pandas2ri.deactivate()


def _move_data(dict_from, dict_to):
    """
    Move all the data from dict_from to dict_to.
    """
    dict_to.clear()
    for k, v in dict_from.items():
        dict_to[k] = v
    dict_from.clear()


def _define_if_given(obj, fn, fn_name_to_set):
    """
    Define `obj.fn_name_to_set` if `fn` is not None. Further, `fn` is called after setting `globalenv` to `obj.globalenv`.
    """
    if fn is None:
        return
    if isinstance(fn, str):
        fn_to_call = getattr(obj.R_module, fn)
    else:

        def fn_to_call(*args, **kwargs):
            return fn(obj.R_module, *args, **kwargs)

    def call_with_separate_global_env(*args, **kwargs):
        global globalenv
        _move_data(obj.globalenv, globalenv)
        res = fn_to_call(*args, **kwargs)
        _move_data(globalenv, obj.globalenv)
        return res

    setattr(obj, fn_name_to_set, call_with_separate_global_env)
