import os
from rpy2.robjects.packages import SignatureTranslatedAnonymousPackage as STAP


class RWrapperMixin:
    def __init__(self, *args, import_base_path, func_name, **kwargs):
        r_files = [f for f in os.listdir(import_base_path) if f.lower().endswith(".r")]
        r_codestring = ""
        for filename in r_files:
            with open(os.path.join(import_base_path, filename), "r") as f:
                r_codestring += f.read()
                r_codestring += "\n"
        module = STAP(r_codestring, "r_predict")
        self.pred_func = getattr(module, func_name)

    def predict(self, stimulus):
        return self.pred_func(*stimulus)

    def act(self, stimulus):
        return self.predict(stimulus)
