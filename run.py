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
    train_data = pd.read_csv(conf.TRAIN_DATA, delimiter=',')
    val_data = pd.read_csv(conf.VAL_DATA, delimiter=',')

    if conf.TRAIN_MODELS is True:
        # Train FAU model
        train_engagement_value = train_data.attention

        train_x_openface_au = load_action_units_x(train_data, "train")
        train_regressor(train_x_openface_au,
                        train_engagement_value, conf.MODEL_AU_NAME)

        # Train FA model
        train_x_openface_face = load_facial_attributes_x(train_data, "train")
        train_regressor(train_x_openface_face,
                        train_engagement_value, conf.MODEL_FACE_NAME)

        # Train BL model
        train_x_openpose = load_body_keypoints_x(train_data, "train")
        train_regressor(train_x_openpose,
                        train_engagement_value, conf.MODEL_BL_NAME)

    if conf.VAL_RESULTS is True:
        val_engagement_value = val_data.attention

        val_x_openface_au = load_action_units_x(val_data, "val")
        val_x_openface_face = load_facial_attributes_x(val_data, "val")
        val_x_openpose = load_body_keypoints_x(val_data, "val")

        au_model = load_model(conf.MODEL_AU_NAME)
        pred_au = au_model.predict(val_x_openface_au)
        evaluate_model(pred_au, val_engagement_value)

        face_model = load_model(conf.MODEL_FACE_NAME)
        pred_face = face_model.predict(val_x_openface_face)
        evaluate_model(pred_face, val_engagement_value)

        bodylandmark_model = load_model(conf.MODEL_BL_NAME)
        pred_bl = bodylandmark_model.predict(val_x_openpose)
        evaluate_model(pred_bl, val_engagement_value)

        average_pred = average_ensemble(pred_au, pred_face, pred_bl)
        evaluate_model(average_pred, val_engagement_value)

        wtd_average_pred = wtd_average_ensemble(pred_au, pred_face, pred_bl)
        evaluate_model(wtd_average_pred, val_engagement_value)
