import pandas as pd
import numpy as np
from scipy.stats import rv_discrete
from ldmunit.models import DADO
from ldmunit.capabilities import BatchTrainable, PredictsLogpdf


class HbayesdmModel(DADO, PredictsLogpdf, BatchTrainable):
    name = "hBayesDM Model"

    def __init__(self, *args, hbayesdm_model_func, col_names, **kwargs):
        assert (
            "inc_postpred" not in kwargs
        ), "'inc_postpred' cannot be specified in the current version"
        DADO.__init__(self, *args, **kwargs)
        self.hbayesdm_model_func = hbayesdm_model_func
        self.col_names = col_names
        self.kwargs = kwargs

    def fit(self, stimuli, actions):
        df = pd.DataFrame(data=stimuli, columns=self.col_names)
        hbayesdm_out = self.hbayesdm_model_func(
            data=df, inc_postpred=True, **self.kwargs
        )
        best_sample_id = np.argmax(hbayesdm_out.par_vals["log_lik"].mean(axis=1))
        self.y_dist = hbayesdm_out.par_vals["y_dist"][best_sample_id]

    def predict(self, stimuli):
        pred_softmax = self.y_dist.reshape(-1, self.y_dist.shape[-1])

        def convert_to_scipy(row):
            return rv_discrete(values=([1, 2], row)).logpmf

        distributions = list(map(convert_to_scipy, pred_softmax))
        return distributions
