"""
Using ZenML patterns to create training pipeline
"""


import random
from typing import Any, Dict, List, Optional
import cv2
import boto3
import pandas as pd
import os
import csv
from sklearn.model_selection import train_test_split
import argparse

from zenml import pipeline
from zenml.logger import get_logger

logger = get_logger(__name__)


from steps import (
    pull_annos_from_labelstudio,
    generate_lst_file,
    generate_rec_file,
    sagemaker_datachannels,
    sagemaker_define_model,
    sagemaker_run_training,
    notify_on_failure,
    notify_on_success
)


@pipeline(on_failure=notify_on_failure)
def tsimlopsdti():
    """
    Model training pipeline.

    This is a pipeline that loads the data, processes it and splits
    it into train and test sets, then search for best hyperparameters,
    trains and evaluates a model.

    Args:
        model_search_space: Search space for hyperparameter tuning
        target_env: The environment to promote the model to
        test_size: Size of holdout set for training 0.0..1.0
        drop_na: If `True` NA values will be removed from dataset
        normalize: If `True` dataset will be normalized with MinMaxScaler
        drop_columns: List of columns to drop from dataset
        min_train_accuracy: Threshold to stop execution if train set accuracy is lower
        min_test_accuracy: Threshold to stop execution if test set accuracy is lower
        fail_on_accuracy_quality_gates: If `True` and `min_train_accuracy` or `min_test_accuracy`
            are not met - execution will be interrupted early
    """
    
    # grab annos from label studio
    
    csv_file_name = os.getcwd() + "\\"+"may15annos.csv"
    iscsv = True

    df = pull_annos_from_labelstudio(iscsv,csv_file_name)
    
    # pull labels, add to dict
    
    # make lst file
    
    lstname = os.getcwd() + "\\"+"tape-exp-test.lst"
    
    generate_lst_file(df, lstname)
    
    # make rec file
    resize_val = 512
    lstlocation = 'tape-exp-test.lst'
    root_folder = r'C:\Users\Administrator\Desktop\notebooks-for-ml-ops\May15-tape-exp-data' #folder where images are stored locally
    rec_name = r'C:\Users\Administrator\Desktop\notebooks-for-ml-ops\tape-exp-test.rec'
    file_name = 'tape-exp-test.rec'
    s3_bucket = "tape-experiment-april6"
    generate_rec_file(resize_val,lstlocation, root_folder, rec_name, file_name, s3_bucket )
    
    # invoke sagemaker
    
    # upload rec files to aws
    
    #define model
    
    # create channels
    
    # kickoff training
    
    last_step = "promote_with_metric_compare"

    notify_on_success(after=[last_step])
