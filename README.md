# Emotiw2020_Engagement_Prediction
This is our submission in the "Eighth Emotion Recognition in the Wild (EmotiW)"" Challenge for the track "Engagement Prediction in the Wild". Our model ended up with 3rd position in the challange.

# TL;DR
We used the weighted average ensemble of three different modalities: facial Action Units(FAU), Facial Attributes(FA) and Body Landmarks(BL) to predict student engagement level. To use our best model for predicting test dataset engagement intensity, run
```bash
virtualenv venv
source venv/bin/activate
pip3 install -r requirements.txt
python3 test.py --model_path pre_trained_models/balanced_models --output_path predicted_engagement
```
The folder predicted_engagement will contain the txt of predicted engagement value on test dataset.

**[Evaluation]
Overall Mean Square Error (Test Dataset): 0.065901**

## Creating virtual environment and installing libraries

### If virtualenv is not installed on system (Ubuntu)
```bash
python3 -m pip install --user virtualenv
```

### Once the virtualenv library gets installed, run the following command to create and activate virtual
```bash
virtualenv venv
source venv/bin/activate
```

#### Installing dependencies
```bash
pip3 install -r requirements.txt
```

## Extracting OpenPose features.
Please refer [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) for the instructions on how to extract COCO Body landmarks. To convert the output of extracted features in our compatible format (from per frame features in JSON to video-level csv feature) and read only required columns(12 Body Landmarks) and frames(clipped from start and end), please use the following command
```bash
python3 extract_openpose_output.py --csv_path <csv_path> --openpose_path <openpose_feat_path> --output_path <path_in_repo_for_results>
```
**(Args):**

*--csv_path*: (required) CSV file containing list of videos and their engagement level.

*--openpose_path*: (required) Path of folder containing OpenPose COCO model's features in raw format.

*--output_path*: (required) Path of the folder to save extracted features.

## How to train models?
Our submission for this track is based on taking a weighted average ensemble of the each model's prediction.
These three models are:
  - FAU : model trained using facial action units
  - FA  : model trained using facial attributes (Gaze, Head Pose, PDM)
  - BL  : model trained using body landmarks extracted using OpenPose

We also propose two different sets of training videos
  - Original training dataset: Only the videos shared for 'training' is used for training models
  - Merged+Balanced dataset: Videos from both the 'validation' as well as 'training' data is used to train the models

```bash
python3 train.py --model_output_path <path_to_save> [--use_original]
```
This command will generate the train and save the three models we are using (FAU, FA and BL).

**(Args):**

*--model_output_path*: (required) Path of the folder where these models will be saved.

*--use_original*: (optional) Train the model using original training dataset. Default is False, in which case Merged+Balanced dataset will be used.


**(Examples):**
#### Training with Merged+Balanced dataset
```bash
python3 train.py --model_output_path pre_trained_models/balanced_models
```

#### Training with original training dataset
```bash
python3 train.py --model_output_path pre_trained_models/original_models --use_original
```

## How to validate models?
To run validation for the models and get evaluation scores, run
```bash
python3 validate.py --model_path <path_of_pre_trained_model>
```
**(Args):**
*--model_path*: (required) Path of the folder where the models (FAU, FA and BL) are saved.

**(Examples):**
```bash
python3 validate.py --model_path pre_trained_models/original_models
```

##### Since our Merged+Balanced dataset uses validation dataset as well for training, running validation on the val_files.csv does not make any sense since the model will be over-fitted in that case. We chose our best performing models trained on original training dataset and their performance on validation dataset as the benchmark for picking models to be re-trained with new partially merged dataset.

## How to generate test results?
To generate txt files containingn prediction of engagement intensity for video, run  
```bash
python3 test.py --model_path <folder_path_containing_models> --output_path <result_path>
```
**(Args):**

*--model_path*: (required) Path of the folder where the models (FAU, FA and BL) are saved.

*--output_path*: (required) Folder to store generated predictions.


**(Examples):**
```bash
python3 test.py --model_path pre_trained_models/balanced_models --output_path predicted_engagement
```
