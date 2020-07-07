"""
Utils.
"""

import numpy as np
import pandas as pd

from sklearn import metrics
from settings import conf


def get_moment_param(segmented_features):
    """Returns mean and std."""
    feat_mean = np.mean(segmented_features, axis=1)
    feat_std = np.std(segmented_features, axis=1)
    return np.concatenate((feat_mean, feat_std), axis=1)


def get_levelwise_mse(pred, trueval):
    """Print level wise mse."""
    feat = pd.DataFrame(
        np.concatenate(
            (pred.reshape(-1, 1), trueval.to_numpy().reshape(-1, 1)), axis=1),
        columns=["pred", "orig"])

    print("level", "mse")

    data = []
    for lev in conf.LEVELS:
        level_wise_feat = feat[feat["orig"] == lev]
        data.append(
            round(metrics.mean_squared_error(level_wise_feat["pred"], level_wise_feat["orig"]), 4))
    return data
