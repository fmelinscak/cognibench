from oct2py import Oct2Py


class OctaveWrapperMixin:
    def __init__(self, *args, import_base_path, func_name, **kwargs):
        self.octave_session = Oct2Py()
        self.octave_session.eval("pkg load statistics")
        self.octave_session.eval(f'addpath("{import_base_path}");')
        self.func_name = func_name

    def predict(self, stimulus):
        return self.octave_session.feval(self.func_name, *stimulus)

    def act(self, stimulus):
        return self.predict(stimulus)
