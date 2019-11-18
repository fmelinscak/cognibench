from importlib import import_module
from ldmunit.models import CACO
from ldmunit.models.wrappers import (
    OctaveWrapperMixin,
    RWrapperMixin,
)


class BEASTsdPython(CACO):
    name = "BEASTsdPython"

    def __init__(self, *args, import_base_path, **kwargs):
        fn_name = "CPC18_BEASTsd_pred"
        module = import_module(f"{import_base_path}.{fn_name}")
        self.pred_fn = getattr(module, fn_name)
        super().__init__(*args, **kwargs)

    def predict(self, *args, **kwargs):
        return self.pred_fn(*args, **kwargs)


class BEASTsdOctave(OctaveWrapperMixin, CACO):
    name = "BEASTsdOctave"

    def __init__(self, *args, import_base_path, **kwargs):
        OctaveWrapperMixin.__init__(
            self,
            import_base_path=import_base_path,
            reset_fn=lambda oct_sess: oct_sess.eval("pkg load statistics;"),
            predict_fn="CPC18_BEASTsd_pred",
        )
        CACO.__init__(self, *args, **kwargs)
        self.reset()


class BEASTsdR(RWrapperMixin, CACO):
    name = "BEASTsdR"

    def __init__(self, *args, import_base_path, **kwargs):
        RWrapperMixin.__init__(
            self, import_base_path=import_base_path, predict_fn="CPC18_BEASTsd_pred"
        )
        CACO.__init__(self, *args, **kwargs)
