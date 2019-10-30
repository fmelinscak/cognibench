from importlib import import_module


class PythonWrapperMixin:
    def __init__(self, *args, import_base_path, func_name, **kwargs):
        module_identifier = f"{import_base_path}.{func_name}"
        module = import_module(module_identifier)
        self.pred_func = getattr(module, func_name)

    def predict(self, stimulus):
        return self.pred_func(*stimulus)

    def act(self, stimulus):
        return self.predict(stimulus)
