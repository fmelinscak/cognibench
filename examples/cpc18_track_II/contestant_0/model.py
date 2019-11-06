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
    def __init__(self):
        self.game_id_idx = stimuli_columns.index("GameID")
        self.block_idx = stimuli_columns.index("block")

    def fit(self, stimuli, actions):
        cols = stimuli_columns + action_columns
        data = np.c_[stimuli, actions]
        df = pd.DataFrame(data, columns=cols)
        df = df[["SubjID", "GameID", "block", "B"]].astype(
            {"SubjID": "int32", "GameID": "int32", "block": "int32", "B": "float32"}
        )
        self.avg_bin_train = (
            df.groupby(["GameID", "block"]).mean()["B"].reset_index().values
        )
        self.overall_mean = np.mean(self.avg_bin_train[:, -1])

    def predict(self, stimuli):
        return [self.pred_one(s) for s in stimuli]

    def pred_one(self, stimulus):
        game_id = int(stimulus[self.game_id_idx])
        block_id = int(stimulus[self.block_idx])
        cell = self.avg_bin_train[
            (self.avg_bin_train[:, 0] == game_id)
            & (self.avg_bin_train[:, 1] == block_id),
            -1,
        ]
        if cell.size == 0:
            return self.overall_mean
        else:
            assert cell.size == 1
            return cell[0]

    # avg_bin_train = train_data.groupby(['GameID', 'block']).mean()['B'].reset_index()
    # test_with_preds = test_data.merge(avg_bin_train, on=['GameID', 'block'])
