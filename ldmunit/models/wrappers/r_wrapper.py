import os
import types
from functools import partial
from rpy2.robjects.packages import SignatureTranslatedAnonymousPackage as STAP
from rpy2.robjects import numpy2ri, pandas2ri, globalenv
from rpy2.robjects.environments import Environment as REnv


def _move_data(dict_from, dict_to):
    dict_to.clear()
    for k, v in dict_from.items():
        dict_to[k] = v
    dict_from.clear()


def _define_if_given(obj, fn, fn_name_to_set):
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


class RWrapperMixin:
    @classmethod
    def activate_conversions(cls, activate_numpy=True, activate_pandas=True):
        if activate_numpy:
            numpy2ri.activate()
        if activate_pandas:
            pandas2ri.activate()

    @classmethod
    def deactivate_conversions(cls, deactivate_numpy=True, deactivate_pandas=True):
        if deactivate_numpy:
            numpy2ri.deactivate()
        if deactivate_pandas:
            pandas2ri.deactivate()

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
        self.globalenv = REnv()
        RWrapperMixin.activate_conversions()

        # source all R files in the given directory, and create an R module from the merged sourcecode
        r_files = [f for f in os.listdir(import_base_path) if f.lower().endswith(".r")]
        r_codestring = ""
        for filename in r_files:
            with open(os.path.join(import_base_path, filename), "r") as f:
                r_codestring += f.read()
                r_codestring += "\n"
        self.R_module = STAP(r_codestring, "r_module")

        _define_if_given(self, reset_fn, "reset")
        _define_if_given(self, predict_fn, "predict")
        _define_if_given(self, fit_fn, "fit")
        _define_if_given(self, update_fn, "update")
        _define_if_given(self, act_fn, "act")
