"""
Train models.
"""

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from scipy.stats import pearsonr

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
    if not os.path.exists(path):
        os.makedirs(path)
    with open("{}/{}.txt".format(path, video_name), "w+") as outfile:
        outfile.write(str(engagement_pred))


def load_model(model_name, model_path=conf.MODEL_PATH):
    """Loads model from desired path."""
    path = "{}/{}".format(model_path, model_name)
    return joblib.load(path)


def save_model(model, model_name, model_path=conf.MODEL_PATH):
    """Save model to desired path."""
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    path = "{}/{}".format(model_path, model_name)
    joblib.dump(model, path)


def train_regressor(train_x, train_y, model_name,
                    model_path=conf.MODEL_PATH, n_trees=200, random_state=50):
    """Train model."""
    rf_regressor = RandomForestRegressor(
        n_estimators=n_trees,
        random_state=random_state
    )
    rf_regressor.fit(train_x, train_y)
    save_model(rf_regressor, model_name, model_path)


def average_ensemble(pred_au, pred_face, pred_bl):
    """Calculated average prediction of three models."""
    video_count = pred_au.shape[0]
    ensemble_data = np.zeros((video_count, 3))
    ensemble_data[:, 0] = pred_au
    ensemble_data[:, 1] = pred_face
    ensemble_data[:, 2] = pred_bl

    avg_pred = np.average(ensemble_data, axis=1)
    return avg_pred


def wtd_average_ensemble(pred_au, pred_face, pred_bl):
    """Calculated weighted prediction of three models."""
    total_wts = (conf.FAU_WEIGHT + conf.FA_WEIGHT + conf.BL_WEIGHT)
    fau_wt = conf.FAU_WEIGHT / total_wts
    fa_wt = conf.FA_WEIGHT / total_wts
    bl_wt = conf.BL_WEIGHT / total_wts

    wtd_pred = (pred_au * fau_wt) + (pred_face * fa_wt) + (pred_bl * bl_wt)

    return wtd_pred


def evaluate_model(pred_y, true_y):
    """Returns mean squared error."""
    mse = round(
        metrics.mean_squared_error(pred_y, true_y), 4)

    pcc = round(pearsonr(pred_y, true_y)[0], 2)

    return mse, pcc
