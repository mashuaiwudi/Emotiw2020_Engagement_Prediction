"""
Extract 12 Body Landmarks from openpose per frame feature output.
"""

import json
import glob
import pandas as pd
import argparse

from settings import conf, openpose as op_conf


def get_main_subject(people_list):
    """Extract main face from list of people returned in openpose output."""
    max_confidence = 0
    est_main_subject = None

    for person in people_list:
        pose_keypoints_2d = person['pose_keypoints_2d']
        pose_keypoint = pd.DataFrame(
            [pose_keypoints_2d], columns=op_conf.COCO_ANNOTATION)
        confidence = pose_keypoint[op_conf.CONF_ANNOTATION].sum(
            axis=1).iloc[0] / len(op_conf.CONF_ANNOTATION)
        if confidence >= max_confidence:
            max_confidence = confidence
            est_main_subject = person
    return est_main_subject


def read_useful_feature(openpose_feat_path):
    """Read openpose output and returns COCO body landmarks of main subject."""
    with open(openpose_feat_path) as f:
        openpose_feat = json.loads(f.read())
    len_person = len(openpose_feat["people"])

    if len_person == 0:
        return None

    if len_person > 1:
        subject = get_main_subject(openpose_feat["people"])
    else:
        subject = openpose_feat["people"][0]

    return subject['pose_keypoints_2d']


def read_openpose_features(video_name, openpose_output_path, output_path):
    """Extract video level feature."""
    video_json_src = "{}/{}".format(openpose_output_path, video_name)
    print(video_json_src)
    allImagePaths = glob.glob("{src}/*".format(src=video_json_src))
    allImagePaths.sort(key=lambda x: int(x.split("_")[-2].split(".")[0]))
    print(len(allImagePaths))
    per_frame_feat = [*map(read_useful_feature, allImagePaths)]
    per_frame_feat = list(filter(None, per_frame_feat))
    per_frame_feat = pd.DataFrame(
        per_frame_feat, columns=op_conf.COCO_ANNOTATION)
    per_frame_feat.to_csv(
        r'{}/{}.txt'.format(
            output_path,
            video_name),
        header=op_conf.COCO_ANNOTATION,
        sep=',',
        mode='w',
        index=False
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process OpenPose output.')

    parser.add_argument('--csv_path', required=True,
                        help='Path of csv containing list of video files.')
    parser.add_argument('--openpose_path', required=True,
                        help='Path of json files extracted using openpose.')
    parser.add_argument('--output_path', required=True,
                        help='Path for extracted 12 body landmarks.')

    args = parser.parse_args()

    data_df = pd.read_csv(args.csv_path, delimiter=",")
    for vid_name in data_df.video_name:
        read_openpose_features(
            vid_name, args.openpose_path, args.output_path)
