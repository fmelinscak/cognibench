from ldmunit.models import CACO
from ldmunit.models.wrappers import (
    PythonWrapperMixin,
    OctaveWrapperMixin,
    RWrapperMixin,
)


class BEASTsdPython(PythonWrapperMixin, CACO):
    name = "BEASTsdPython"

    def __init__(self, *args, import_base_path, **kwargs):
        PythonWrapperMixin.__init__(
            self, import_base_path=import_base_path, func_name="CPC18_BEASTsd_pred"
        )
        CACO.__init__(self, *args, **kwargs)


class BEASTsdOctave(OctaveWrapperMixin, CACO):
    name = "BEASTsdOctave"

    def __init__(self, *args, import_base_path, **kwargs):
        OctaveWrapperMixin.__init__(
            self, import_base_path=import_base_path, func_name="CPC18_BEASTsd_pred"
        )
        CACO.__init__(self, *args, **kwargs)


class BEASTsdR(RWrapperMixin, CACO):
    name = "BEASTsdR"

    def __init__(self, *args, import_base_path, **kwargs):
        RWrapperMixin.__init__(
            self, import_base_path=import_base_path, func_name="CPC18_BEASTsd_pred"
        )
        CACO.__init__(self, *args, **kwargs)
