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
import sagemaker

from utils.constants import CSV_FILE_NAME, LST_FILE_NAME
from uuid import UUID


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


from zenml import pipeline
#from zenml.client import Client
from zenml.logger import get_logger
from zenml import get_pipeline_context

from zenml import ArtifactConfig, get_step_context, step
from zenml.client import Client
from zenml.integrations.mlflow.experiment_trackers import (
    MLFlowExperimentTracker,
)
from zenml.integrations.mlflow.steps.mlflow_registry import (
    mlflow_register_model_step,
)
logger = get_logger(__name__)
client = Client()
experiment_tracker = client.active_stack.experiment_tracker

if not experiment_tracker or not isinstance(
    experiment_tracker, MLFlowExperimentTracker
):
    raise RuntimeError(
        "Your active stack needs to contain a MLFlow experiment tracker for "
        "this example to work."
    )


#@pipeline(on_failure=notify_on_failure, experiment_tracker=experiment_tracker.name)
@pipeline(on_failure=notify_on_failure)
def training_pipeline(
    train_dataset_id: Optional[UUID] = None,
    test_dataset_id: Optional[UUID] = None,
):
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
    

    
    dataset_trn = client.get_artifact_version(
        name_id_or_prefix=train_dataset_id
        )
    dataset_tst = client.get_artifact_version(
        name_id_or_prefix=test_dataset_id
        )
    
    # 1. grab annos from label studio
    
    csv_file_name = CSV_FILE_NAME
    iscsv = True # use this before directly connecting label studio

    df = pull_annos_from_labelstudio(iscsv,csv_file_name)
    
    # pull labels, add to dict
    # @TODO: Need to flesh this out :)
    
    # 2. make lst file
    
    lstname = LST_FILE_NAME
    
    generate_lst_file(df, lstname)
    
    # 3. make rec file
    
    resize_val = 512
    lstlocation = 'tape-exp-test.lst'
    root_folder = r'C:\Users\Administrator\Desktop\notebooks-for-ml-ops\May15-tape-exp-data' #folder where images are stored locally
    rec_name = r'C:\Users\Administrator\Desktop\notebooks-for-ml-ops\tape-exp-test.rec'
    file_name = 'tape-exp-test.rec'
    s3_bucket = "tape-experiment-april6"
    generate_rec_file(resize_val,lstlocation, root_folder, rec_name, file_name, s3_bucket )
    
    # 4. invoke sagemaker, upload rec files to aws
    
    bucket = 'tubes-tape-exp-models'
    prefix = 'retinanet'
    sess = sagemaker.Session()
    model_bucket_path = "s3://tubes-tape-exp-models/"
    s3_output_location = "s3://{}/{}/output".format(bucket,prefix)

    training_image, data_channels = sagemaker_datachannels(sess,rec_name,bucket,prefix,model_bucket_path,s3_bucket)
    
    # 5. define model
    
    try:
        role = sagemaker.get_execution_role()
    except ValueError:
        iam = boto3.client('iam')
        role = iam.get_role(RoleName='SkylerSageMakerRole')['Role']['Arn']
    region = 'us-east-1'

    print('Sagemaker session :', sess)
    print('default bucket :', bucket)
    print('Prefix :', prefix)
    print('Region selected :', region)
    print('IAM role :', role)
    
    inst_type = "ml.p3.8xlarge" #<---costly, but fast. 21 mins per 100 epochs for 500img dataset
    
    od_mdl = sagemaker_define_model(sess, role, inst_type, training_image, s3_output_location)

    # 6. kickoff training
    
    sagemaker_run_training(od_mdl,data_channels)
    
    notify_on_success(after=["sagemaker_run_training"])
