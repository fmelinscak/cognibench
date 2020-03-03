import numpy as np
import pandas as pd
import ast


def from_np_array(array_string):
    """
    Construct a numpy.ndarray from its string representation.
    """
    array_string = ",".join(array_string.replace("[ ", "[").split())
    return np.array(ast.literal_eval(array_string))


def get_observation_from_idx(filepath, idx, np_array=False):
    """
    Read observation for a single index.
    """
    if np_array:
        df = pd.read_csv(filepath, converters={"stimuli": from_np_array})
    else:
        df = pd.read_csv(filepath)
    assert idx in df["sub"].values, "subject index not in dataframe"
    res = df.loc[df["sub"] == idx]
    res = res.loc[:, ["rewards", "actions", "stimuli"]]
    out = res.to_dict(orient="list")
    return out


def get_simulation_data(filepath, n_sub, np_array=False):
    """
    Read example simulation data files.
    """
    res = []
    for i in range(n_sub):
        df_dict = get_observation_from_idx(filepath, i, np_array=np_array)
        res.append(df_dict)

    return res


def get_model_params(filepath):
    """
    Read example model parameter files. If integer representation of
    a parameter is equal to itself, then the parameter is stored as
    integer instead of float.
    """
    df = pd.read_csv(filepath)
    paras_list = []
    col_names = df.columns
    values = df.values
    for i, row in enumerate(values):
        par = dict()
        for j, key in enumerate(col_names):
            val = row[j]
            if int(val) == val:
                val = int(val)
            par[key] = val
        paras_list.append(par)

    return paras_list
