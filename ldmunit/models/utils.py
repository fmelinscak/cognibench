import gym
import types
from functools import partial
import sciunit
import numpy as np
from ldmunit.capabilities import Interactive, MultiSubjectModel


def multi_from_single_cls(single_cls):
    """
    Create a multi-subject model class from a single-subject model class.

    Parameters
    ----------
    single_cls : :class:`ldmunit.model.LDMModel`
        A single-subject model class.

    Returns
    -------
    out_cls : :class:`ldmunit.model.LDMModel`
        A multi-subject model class. Each method of the new class now takes an additional subject index (0-based) as
        their first argument. If a subject index is not provided during a method call, the method of subject with index
        0 is called by default.
    """
    multi_cls_name = "Multi" + single_cls.__name__
    return MultiMeta(
        multi_cls_name,
        (single_cls,),
        {"name": single_cls.name, "__doc__": single_cls.__doc__,},
    )


def single_from_multi_obj(model, subj_idx):
    """
    Temporarily convert a multi-subject model object created by multi_from_single_cls to a single-subject object
    for the given subject index. The model returned by this function behaves as a single-subject model where the subject
    is given by `subj_idx`. The multi-subject variants of the replaced methods are stored with multi suffix to be restored
    later.

    Parameters
    ----------
    model : :class:`ldmunit.models.LDMModel`, :class:`ldmunit.capabilities.MultiSubjectModel`
        Multi-subject model.

    subj_idx : int
        0-based subject index. The returned model will be a single-subject model of the subject with this index.

    Returns
    -------
    out_model : :class:`ldmunit.models.LDMModel`, :class:`ldmunit.capabilities.MultiSubjectModel`
        A proxy single-subject model object that delegates all method calls to the subject model with the given index.

    See Also
    --------
    :py:func:`reverse_single_from_multi_obj`
    """
    assert isinstance(model, MultiSubjectModel)

    def make_new_fn(old_fn):
        def new_fn(self, *args, **kwargs):
            return old_fn(subj_idx, *args, **kwargs)

        return new_fn

    for fn_name in model.multi_subject_methods:
        old_fn = getattr(model, fn_name)
        new_fn = make_new_fn(old_fn)
        setattr(model, f"{fn_name}_multi", old_fn)
        setattr(model, fn_name, new_fn.__get__(model))
    return model


def reverse_single_from_multi_obj(model):
    """
    Reverse a single from multi object conversion performed by `single_from_multi_obj`.

    Parameters
    ----------
    model : :class:`ldmunit.models.LDMModel`, :class:`ldmunit.capabilities.MultiSubjectModel`
        A proxy single-subject model object that delegates all method calls to the subject model with the given index.

    Returns
    -------
    out_model : :class:`ldmunit.models.LDMModel`, :class:`ldmunit.capabilities.MultiSubjectModel`
        Original multi-subject model.

    See Also
    --------
    :py:func:`single_from_multi_obj`
    """
    for fn_name in model.multi_subject_methods:
        multi_name = f"{fn_name}_multi"
        old_fn = getattr(model, multi_name)
        setattr(model, fn_name, old_fn)
        delattr(model, multi_name)
    return model


class MultiMeta(type):
    """
    MultiMeta is a metaclass for creating multi-subject model classes from single-subject model classes.

    Each method of the returned class takes an additional subject
    index as their first argument. This index is used to select the individual
    single-subject model to use. In this regard, the returned class is semantically
    similar to a list of single-subject models while also satisfying model class capability requirements. In addition,
    the returned model class now additionally inherits from `ldmunit.capabilities.MultiSubjectModel`, which is required
    by parts of `ldmunit` that expect multi-subject models as input.

    In order to not break compatibility with code that doesn't use subject indices, the returned class forwards a method
    call to the subject model with index 0 if no index is passed. Users should not rely on this behaviour as it is only
    provided for compatibility with certain parts of sciunit library (e.g. sciunit calls `describe` method of the model
    with no subject index when run on a jupyter notebook).

    This metaclass is not intended to be used directly by `ldmunit` users. Users should use
    multi_from_single_cls for automatically generating multi-subject models from single-subject ones without having to
    bother with implementation details.

    See Also
    --------
    :py:func:`multi_from_single_cls`
    """

    def __new__(cls, name, bases, dct):
        single_cls = bases[0]
        base_classes = single_cls.__bases__ + (MultiSubjectModel,)
        out_cls = super().__new__(cls, name, base_classes, dct)
        # TODO: implement this in a better way
        methods_to_define = [
            f
            for f in dir(single_cls)
            if callable(getattr(single_cls, f))
            and not (f.startswith("__") and f.endswith("__"))
        ]

        def multi_init(self, *args, n_subj, **kwargs):
            self.subject_models = []
            for _ in range(n_subj):
                self.subject_models.append(single_cls(*args, **kwargs))
            self.n_subjects = len(self.subject_models)

            def new_fn(*args, fn_name, **kwargs):
                if len(args) == 0:
                    idx = 0
                else:
                    idx = args[0]
                    args = args[1:]
                return getattr(self.subject_models[idx], fn_name)(*args, **kwargs)

            for fn_name in methods_to_define:
                setattr(out_cls, fn_name, partial(new_fn, fn_name=fn_name))

        def fit_jointly(self, *args, **kwargs):
            """
            Default implementation simply fits each subject model separately. In case you need more complex behaviour,
            such as hierarchical joint fitting of the subject models, you can

            1. define your separate multi-subject model, or
            2. subclass the output of this metaclass and override `fit_jointly` to perform the
            desired fitting procedure.

            Parameters
            ----------
            args : iterable
                Each positional argument to this function must be an iterable that contains the subject-specific
                fitting arguments.
            kwargs : dict
                Each keyword argument to this function must be an iterable that contains the subject-specific fitting
                keyword arguments.
            """
            # TODO: provide an example of item (2) above.
            for i, model in enumerate(self.subject_models):
                curr_args = []
                curr_kwargs = dict()
                for arg in args:
                    curr_args.append(arg[i])
                for k, v in kwargs.items():
                    curr_kwargs[k] = v[i]
                model.fit(*curr_args, **curr_kwargs)

        out_cls.__init__ = multi_init
        out_cls.fit_jointly = fit_jointly
        out_cls.multi_subject_methods = methods_to_define

        return out_cls
