import pandas as pd
import numpy as np
from scipy.stats import rv_discrete
from ldmunit.models import LDMModel
from ldmunit.capabilities import DiscreteAction, DiscreteObservation
from ldmunit.capabilities import PredictsLogpdf, ReturnsNumParams


class HbayesdmModel(
    LDMModel, DiscreteAction, DiscreteObservation, PredictsLogpdf, ReturnsNumParams
):
    name = "hBayesDM Model"

    def __init__(
        self, *args, hbayesdm_model_func, col_names, n_action, n_obs, **kwargs
    ):
        assert (
            "inc_postpred" not in kwargs
        ), "'inc_postpred' cannot be specified in the current version"
        LDMModel.__init__(self, *args, **kwargs)
        self.set_action_space(n_action)
        self.set_observation_space(n_obs)
        self.hbayesdm_model_func = hbayesdm_model_func
        self.col_names = col_names
        self.kwargs = kwargs

    def n_params(self):
        try:
            return self.n_hbayesdm_paras
        except AttributeError:
            print("self.n_hbayesdm_paras not defined; try fitting the model first.")

    def fit(self, stimuli, actions):
        df = pd.DataFrame(data=stimuli, columns=self.col_names)
        hbayesdm_out = self.hbayesdm_model_func(
            data=df, inc_postpred=True, **self.kwargs
        )
        self.n_hbayesdm_paras = len(hbayesdm_out.parameters)
        best_sample_id = np.argmax(hbayesdm_out.par_vals["log_lik"].mean(axis=1))
        self.y_dist = hbayesdm_out.par_vals["y_dist"][best_sample_id]

    def predict(self, stimuli):
        pred_softmax = self.y_dist.reshape(-1, self.y_dist.shape[-1])

        def convert_to_scipy(row):
            return rv_discrete(values=(np.arange(1, 5), row)).logpmf

        distributions = list(map(convert_to_scipy, pred_softmax))
        return distributions
