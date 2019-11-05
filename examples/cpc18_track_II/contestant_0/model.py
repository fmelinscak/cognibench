import numpy as np
import pandas as pd


# TODO: should be defined in dataset API part of the project
stimuli_columns = [
    "SubjID",
    "Location",
    "Gender",
    "Age",
    "Order",
    "GameID",
    "Ha",
    "pHa",
    "La",
    "LotShapeA",
    "LotNumA",
    "Hb",
    "pHb",
    "Lb",
    "LotShapeB",
    "LotNumB",
    "Amb",
    "Corr",
    "block",
    "Feedback",
    "Dom",
    "diffUV",
    "RatioMin",
    "SignMax",
    "pBbet_Unbiased1",
    "pBbet_UnbiasedFB",
    "pBbet_Uniform",
    "pBbet_Sign1",
    "pBbet_SignFB",
    "diffBEV0",
    "diffBEVfb",
    "diffMins",
    "diffSignEV",
    "diffEV",
    "diffMaxs",
    "diffSDs",
]
action_columns = ["B"]


class Model:
    def fit(self, stimuli, actions):
        cols = stimuli_columns + action_columns
        data = np.c_[stimuli, actions]
        df = pd.DataFrame(data, columns=cols)
        self.avg_bin_train = df.groupby(["GameID", "block"]).mean()["B"].reset_index()

    def predict(self, stimuli):
        df = pd.DataFrame(stimuli, columns=stimuli_columns)
        df_preds = df.merge(self.avg_bin_train, on=["GameID", "block"])
        return df_preds["B"].values

    # avg_bin_train = train_data.groupby(['GameID', 'block']).mean()['B'].reset_index()
    # test_with_preds = test_data.merge(avg_bin_train, on=['GameID', 'block'])
