import os

# Dataset export from labelstudio
LABELED_DATASET_NAME = "ship_od_dataset"

# Trained Model
TRAINED_MODEL_NAME = "Trained_YOLO"

# Name of Model in ZenML Model Control Plane
ZENML_MODEL_NAME = "tsi-mlops"

# Constants for inference pipeline
PREDICTIONS_DATASET_ARTIFACT_NAME = "predictions_dataset_json"
DATASET_NAME = "ships"
DATASET_DIR = "data/ships/subset"

CSV_FILE_NAME = os.getcwd() + "\\"+"may15annos.csv"
LST_FILE_NAME = os.getcwd() + "\\"+"tape-exp-test.lst"