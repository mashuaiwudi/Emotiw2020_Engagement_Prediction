"""
Load features.
"""

import os
import numpy as np
import pandas as pd

from settings import conf
import utils


def read_openface_features(video_name, data_type, required_feat):
    """Read list of required facial feat from extracted openface features."""
    src = conf.OPENFACE_PATH[data_type]
    video_feature_src = "{}/{}.txt".format(src, video_name)
    if not os.path.exists(video_feature_src):
        video_feature_src = video_feature_src.replace(".txt", "")

    openface_feat = pd.read_csv(
        video_feature_src,
        delimiter=", ",
        engine='python',
        skipinitialspace=True
    ).fillna(0)

    filtered_feat = openface_feat[
        (openface_feat["timestamp"] > conf.START_TIME_SEC) &
        (openface_feat["timestamp"] < conf.END_TIME_SEC) &
        (openface_feat["success"] == 1)
    ]
    selected_openface_feat = filtered_feat[required_feat]
    selected_openface_feat = selected_openface_feat.to_numpy()
    final_feat = utils.get_moment_param([selected_openface_feat])

    return final_feat


def read_openpose_features(video_name, data_type, required_feat):
    """Read list of required body keypoints from extracted openpose features."""
    src = conf.OPENPOSE_PATH[data_type]
    video_feature_src = "{}/{}.txt".format(src, video_name)

    openpose_feat = pd.read_csv(
        video_feature_src,
        delimiter=","
    )
    openpose_feat = openpose_feat[300:-700]

    openpose_feat = openpose_feat[required_feat]
    selected_openpose_feat = openpose_feat.to_numpy()
    final_feat = utils.get_moment_param([selected_openpose_feat])

    return final_feat


def load_action_units_x(img_df, data_type):
    """Load the required feature columns from extracted openface features."""
    train_x = []
    for vid_name in img_df.video_name:
        train_x.append(
            read_openface_features(vid_name, data_type, conf.OPENFACE_MODALITY_AU))
    train_x = np.asarray(train_x)
    train_x = train_x.reshape(train_x.shape[0], -1)

    return train_x


def load_facial_attributes_x(img_df, data_type):
    """Load the required feature columns from extracted openface features."""
    train_x = []
    for vid_name in img_df.video_name:
        train_x.append(
            read_openface_features(vid_name, data_type, conf.OPENFACE_MODALITY_FACE))
    train_x = np.asarray(train_x)
    train_x = train_x.reshape(train_x.shape[0], -1)

    return train_x


def load_body_keypoints_x(img_df, data_type):
    """Load the required feature columns from extracted openpose features."""
    train_x = []
    for vid_name in img_df.video_name:
        train_x.append(
            read_openpose_features(vid_name, data_type, conf.OPENPOSE_MODALITY))
    train_x = np.asarray(train_x)
    train_x = train_x.reshape(train_x.shape[0], -1)

    return train_x
