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


def save_output(path, video_name_df, pred):
    """Save test predictions."""
    output_df = pd.DataFrame([video_name_df, pred]).T
    output_df.columns = ['A', 'B']
    output_df.apply(lambda row: write_file(
        path, row['A'], row['B']), axis=1)


def write_file(path, video_name, engagement_pred):
    with open("{}/{}.txt".format(path, video_name), "w+") as outfile:
        outfile.write(str(engagement_pred))
