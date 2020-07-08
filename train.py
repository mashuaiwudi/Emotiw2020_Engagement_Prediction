"""
Main run file.
"""

import pandas as pd
import argparse
from settings import conf
from load_data import load_action_units_x, load_facial_attributes_x,\
    load_body_keypoints_x
from utils import train_regressor

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process OpenPose output.')

    parser.add_argument('--model_output_path', required=True,
                        help='Path of folder containing models.')
    parser.add_argument('--use_original', action='store_true')

    args = parser.parse_args()
    model_path = args.model_output_path

    train_data_path = conf.TRAIN_DATA_MERGED
    data_type = "merged"

    if args.use_original:
        train_data_path = conf.TRAIN_DATA
        data_type = "train"

    train_data = pd.read_csv(train_data_path, delimiter=',')
    print("Train data: ", train_data.shape)

    print("Training models")
    # Train FAU model
    train_engagement_value = train_data.attention

    train_x_openface_au = load_action_units_x(train_data, data_type)
    print("Loaded FAU features")
    train_regressor(train_x_openface_au,
                    train_engagement_value, conf.MODEL_AU_NAME, model_path)
    print("Trained FAU model")

    # Train FA model
    train_x_openface_face = load_facial_attributes_x(train_data, data_type)
    print("Loaded FA features")
    train_regressor(train_x_openface_face,
                    train_engagement_value, conf.MODEL_FACE_NAME, model_path)
    print("Trained FA model")

    # Train BL model
    train_x_openpose = load_body_keypoints_x(train_data, data_type)
    print("Loaded BL features")
    train_regressor(train_x_openpose,
                    train_engagement_value, conf.MODEL_BL_NAME, model_path)
    print("Trained BL model")
