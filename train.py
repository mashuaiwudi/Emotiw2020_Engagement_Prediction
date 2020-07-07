"""
Train models.
"""

import joblib
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from settings import conf
from sklearn import metrics


def load_model(model_name, model_path=conf.MODEL_PATH):
    """Loads model from desired path."""
    path = "{}/{}".format(model_path, model_name)
    return joblib.load(path)


def save_model(model, model_name, model_path=conf.MODEL_PATH):
    """Save model to desired path."""
    path = "{}/{}".format(model_path, model_name)
    joblib.dump(model, path)


def train_regressor(train_x, train_y, model_name, n_trees=200, random_state=50):
    """Train model."""
    rf_regressor = RandomForestRegressor(
        n_estimators=n_trees,
        random_state=random_state
    )
    rf_regressor.fit(train_x, train_y)
    save_model(rf_regressor, model_name, conf.MODEL_PATH)


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
        metrics.mean_absolute_error(pred_y, true_y), 4)
    print("MSE: ", mse)

    return mse
