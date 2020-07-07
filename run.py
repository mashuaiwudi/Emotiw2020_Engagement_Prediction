"""
Main run file.
"""

import pandas as pd

from settings import conf
from processing.load_data import load_action_units_x, load_facial_attributes_x,\
    load_body_keypoints_x
from train import train_regressor, load_model, evaluate_model, \
    average_ensemble, wtd_average_ensemble


if __name__ == "__main__":

    if conf.TRAIN_MODELS is True:
        train_data = pd.read_csv(conf.TRAIN_DATA, delimiter=',')
        print("Training models")
        # Train FAU model
        train_engagement_value = train_data.attention

        train_x_openface_au = load_action_units_x(train_data, "train")
        print("Loaded FAU features")
        train_regressor(train_x_openface_au,
                        train_engagement_value, conf.MODEL_AU_NAME)
        print("Trained FAU model")

        # Train FA model
        train_x_openface_face = load_facial_attributes_x(train_data, "train")
        print("Loaded FA features")
        train_regressor(train_x_openface_face,
                        train_engagement_value, conf.MODEL_FACE_NAME)
        print("Trained FA model")

        # Train BL model
        train_x_openpose = load_body_keypoints_x(train_data, "train")
        print("Loaded BL features")
        train_regressor(train_x_openpose,
                        train_engagement_value, conf.MODEL_BL_NAME)
        print("Trained BL model")

    if conf.VAL_RESULTS is True:
        val_data = pd.read_csv(conf.VAL_DATA, delimiter=',')
        val_engagement_value = val_data.attention

        print("Validating models")

        val_x_openface_au = load_action_units_x(val_data, "val")
        print("Loaded validation FAU features")
        val_x_openface_face = load_facial_attributes_x(val_data, "val")
        print("Loaded validation FA features")
        val_x_openpose = load_body_keypoints_x(val_data, "val")
        print("Loaded validation BL features")

        au_model = load_model(conf.MODEL_AU_NAME)
        pred_au = au_model.predict(val_x_openface_au)
        mse_au, pcc_au = evaluate_model(pred_au, val_engagement_value)
        print("FAU: MSE = {}, PCC = {}".format(mse_au, pcc_au))

        face_model = load_model(conf.MODEL_FACE_NAME)
        pred_face = face_model.predict(val_x_openface_face)
        mse_face, pcc_face = evaluate_model(pred_face, val_engagement_value)
        print("FA: MSE = {}, PCC = {}".format(mse_face, pcc_face))

        bodylandmark_model = load_model(conf.MODEL_BL_NAME)
        pred_bl = bodylandmark_model.predict(val_x_openpose)
        mse_bl, pcc_bl = evaluate_model(pred_bl, val_engagement_value)
        print("BL: MSE = {}, PCC = {}".format(mse_bl, pcc_bl))

        average_pred = average_ensemble(pred_au, pred_face, pred_bl)
        mse_avg, pcc_avg = evaluate_model(average_pred, val_engagement_value)
        print("Avg Ensemble: MSE = {}, PCC = {}".format(mse_avg, pcc_avg))

        wtd_average_pred = wtd_average_ensemble(pred_au, pred_face, pred_bl)
        mse_wtd, pcc_wtd = evaluate_model(
            wtd_average_pred, val_engagement_value)
        print("Wtd Avg Ensemble: MSE = {}, PCC = {}".format(mse_wtd, pcc_wtd))
