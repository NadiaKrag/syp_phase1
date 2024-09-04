import numpy as np
import pandas as pd
import os

def get_most_important_features():
    last_version = 0
    for f in os.listdir("features/"):
        cur_version = int(f.split("_")[0])
        if cur_version >= last_version:
            last_version = cur_version

    features = pd.read_pickle("features/{}_features.pkl".format(last_version))
    actual = pd.read_pickle("features/{}_actual.pkl".format(last_version))
    null = pd.read_pickle("features/{}_null.pkl".format(last_version))

    null = null.groupby(null.index).agg({"gain":["mean","std"]})
    null.columns = null.columns.droplevel()
    null["threshold"] = null["mean"] + null["std"]

    return features.join(actual.loc[actual["gain"] > null["threshold"]],how="right")

