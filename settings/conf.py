"""
    Basic Conf.
"""

from settings import openface as of_conf
from settings import openpose as op_conf


TRAIN_MODELS = False
VAL_RESULTS = False
TEST_RESULTS = True

TRAIN_DATA = "settings/train_files.csv"
TEST_DATA = "settings/test_files.csv"
VAL_DATA = "settings/val_files.csv"

OPENPOSE_PATH = {
    "train":  "openpose_features/train",
    "test":  "openpose_features/test",
    "val":  "openpose_features/validation"}

OPENFACE_PATH = {
    "train": "openface_features/train",
    "test": "openface_features/test",
    "val": "openface_features/validation"
}


FAU_WEIGHT = 1 / (0.054)
FA_WEIGHT = 1 / (0.071)
BL_WEIGHT = 1 / (0.067)


# Settings for subsampling openface_features
START_TIME_SEC = 30
END_TIME_SEC = 270
FRAME_PER_SEC = 10

# Settings for subsampling openpose_features
START_FRAME = 300
END_FRAME = -700


# Levels of engagement intensity
LEVELS = [0, 0.33, 0.66, 1]

OPENFACE_MODALITY_AU = of_conf.AU_INTENSITY_HEADER
OPENFACE_MODALITY_FACE = of_conf.PDM_HEADER_NONRIGID + \
    of_conf.GAZE_HEADER + of_conf.HEAD_POSE_HEADER
OPENPOSE_MODALITY = op_conf.USEFUL_ANNOTATIONS

TEST_OUTPUT_PATH = "test_results"
MODEL_PATH = "pre_trained_models/submission_2"
MODEL_AU_NAME = "model_FAU.sav"
MODEL_FACE_NAME = "model_FA.sav"
MODEL_BL_NAME = "model_BL.sav"
