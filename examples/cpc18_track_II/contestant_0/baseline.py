import os

# os.chdir("---")

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import pandas as pd
from splitLongData import splitLongData
import math
from scipy.stats.stats import pearsonr

# A function used to evaluate predictive performance
def my_evaluate(y, y_hat, do_MSE=True, do_cor=True, do_plot=True):
    rmse = None
    cor_yy = None
    if do_MSE:
        rmse = math.sqrt(mean_squared_error(y, y_hat))
        print("RMSE is {}".format(rmse))

    if do_cor:
        cor_yy = pearsonr(y, y_hat)[0]
        print("correlation is {}".format(cor_yy))

    if do_plot:
        fit = np.polyfit(y_hat, y, deg=1)
        plt.plot(y_hat, fit[0] * y_hat + fit[1], color="black")
        plt.scatter(y_hat, y, color="black")
        y = range(2)
        plt.plot(y, "--", color="red")
        plt.show()

    return rmse, cor_yy


# Transform outputs to be consistent with maximization (i.e., not minimization)
def pBpMaxTransform(orig_vec, is_b_max):
    orig_vec.name = "B"
    new_vec = pd.concat([orig_vec, is_b_max], axis=1)
    new_vec.loc[new_vec["isBMax"] == False, "B"] = 1 - new_vec["B"]
    return new_vec["B"]


def main():
    # Read data
    individual_block_avgs = pd.read_csv("individualBlockAvgs.csv")

    is_b_max = individual_block_avgs["diffEV"] >= 0
    is_b_max.name = "isBMax"
    individual_block_avgs["B"] = pBpMaxTransform(individual_block_avgs["B"], is_b_max)

    # keep only relevant data
    data = individual_block_avgs[["SubjID", "GameID", "block", "B"]]
    data = data.assign(isBMax=is_b_max.values)
    # data['isBMax'] = is_b_max.values

    # split data to train_data and test #
    new_data = data.loc[data.SubjID >= 60000]
    train_data, test_data = splitLongData(new_data, seed=1)
    train_data = pd.concat(
        [train_data, data.loc[data.SubjID < 60000]], ignore_index=True
    )

    ########################
    # Naive baseline: avgs #
    ########################
    my_evaluate(
        pBpMaxTransform(test_with_preds["B_x"], test_with_preds["isBMax"]),
        pBpMaxTransform(test_with_preds["B_y"], test_with_preds["isBMax"]),
    )


if __name__ == "__main__":
    main()
