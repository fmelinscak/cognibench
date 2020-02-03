from os.path import join as pathjoin
import numpy as np
import pandas as pd
from cognibench.models.associative_learning import (
    BetaBinomialModel,
    LSSPDModel,
    RwNormModel,
    KrwNormModel,
    RandomRespondModel,
)

data_folder = "data"
models = [
    (
        pathjoin("data", "multi-beta_binomial_prior.csv"),
        pathjoin("data", "multi-beta_binomial.csv"),
        BetaBinomialModel,
    ),
    (
        pathjoin("data", "multi-krw_norm_prior.csv"),
        pathjoin("data", "multi-krw_norm.csv"),
        KrwNormModel,
    ),
    (
        pathjoin("data", "multi-lsspd_prior.csv"),
        pathjoin("data", "multi-lsspd.csv"),
        LSSPDModel,
    ),
    (
        pathjoin("data", "multi-rr_al_prior.csv"),
        pathjoin("data", "multi-rr_al.csv"),
        RandomRespondModel,
    ),
    (
        pathjoin("data", "multi-rw_norm_prior.csv"),
        pathjoin("data", "multi-rw_norm.csv"),
        RwNormModel,
    ),
]


def conv_stimuli(string):
    return np.array([int(x) for x in string[1:-1].split()])


for prior_file, data_file, model_ctor in models:
    prior_df = pd.read_csv(prior_file)
    data_df = pd.read_csv(data_file, converters={"stimuli": conv_stimuli})
    n_obs = len(data_df.at[0, "stimuli"])
    param_list = prior_df.to_dict("records")
    n_params = len(param_list)
    n_samples_per_subj = len(data_df) // n_params
    for subject_idx, params in enumerate(param_list):
        model = model_ctor(n_obs=n_obs, **params)
        model.reset()
        beg = subject_idx * n_samples_per_subj
        end = (subject_idx + 1) * n_samples_per_subj
        for i in range(beg, end):
            row = data_df.iloc[i]
            action = model.act(row["stimuli"])
            data_df.at[i, "actions"] = action
            model.update(row["stimuli"], row["rewards"], action, False)
    data_df.to_csv(data_file, index=False)
