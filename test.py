"""
Main run file.
"""
import argparse
import pandas as pd
from settings import conf
from load_data import load_action_units_x, load_facial_attributes_x,\
    load_body_keypoints_x
from utils import save_output
from utils import load_model, average_ensemble, wtd_average_ensemble


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process OpenPose output.')

    parser.add_argument('--model_path', required=True,
                        help='Path of folder containing models.')
    parser.add_argument('--output_path', required=True,
                        help='Output folder for avg and wtd_avg predictions.')

    args = parser.parse_args()
    model_path = args.model_path
    test_output_path = args.output_path

    test_data = pd.read_csv(conf.TEST_DATA, delimiter=',')

    print("Generating test results")

    test_x_openface_au = load_action_units_x(test_data, "test")
    print("Loaded test FAU features")
    test_x_openface_face = load_facial_attributes_x(test_data, "test")
    print("Loaded test FA features")
    test_x_openpose = load_body_keypoints_x(test_data, "test")
    print("Loaded test BL features")

    au_model = load_model(conf.MODEL_AU_NAME, model_path)
    face_model = load_model(conf.MODEL_FACE_NAME, model_path)
    bodylandmark_model = load_model(conf.MODEL_BL_NAME, model_path)
    print("Loaded models")

    pred_au = au_model.predict(test_x_openface_au)
    pred_face = face_model.predict(test_x_openface_face)
    pred_bl = bodylandmark_model.predict(test_x_openpose)

    average_pred = average_ensemble(pred_au, pred_face, pred_bl)
    wtd_average_pred = wtd_average_ensemble(pred_au, pred_face, pred_bl)

    # save_output("{}/avg/".format(test_output_path),
    #             test_data.video_name, average_pred)
    # print("Saved average ensemble output")

    save_output("{}/wtd_avg/".format(test_output_path),
                test_data.video_name, wtd_average_pred)
    print("Saved wtd average ensemble output")
