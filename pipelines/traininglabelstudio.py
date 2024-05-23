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
from sagemaker import image_uris
from sagemaker.workflow.pipeline_context import PipelineSession

from utils.constants import CSV_FILE_NAME, LST_FILE_NAME
from uuid import UUID


# from steps import (
#     pull_annos_from_labelstudio,
#     generate_lst_file,
#     generate_rec_file,
#     sagemaker_datachannels,
#     sagemaker_define_model,
#     sagemaker_run_training,
#     notify_on_failure,
#     notify_on_success
# )

from steps import (
    pull_annos_from_labelstudio,
    generate_lst_file,
    generate_rec_file,
    notify_on_failure,
)

from zenml import pipeline
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
    model_search_space: Dict[str, Any],
    target_env: str,
    test_size: float = 0.2,
    drop_na: Optional[bool] = None,
    normalize: Optional[bool] = None,
    drop_columns: Optional[List[str]] = None,
    min_train_accuracy: float = 0.0,
    min_test_accuracy: float = 0.0,
    fail_on_accuracy_quality_gates: bool = False,
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
    #train_dataset_id: Optional[UUID] = None,
    #test_dataset_id: Optional[UUID] = None,

    
    # dataset_trn = client.get_artifact_version(
    #     name_id_or_prefix=train_dataset_id
    #     )
    # dataset_tst = client.get_artifact_version(
    #     name_id_or_prefix=test_dataset_id
    #     )
    
    # 1. grab annos from label studio
    
    csv_file_name = CSV_FILE_NAME
    iscsv = True # use this before directly connecting label studio

    df = pull_annos_from_labelstudio(iscsv,csv_file_name)
    
    print(df)
    
    # pull labels, add to dict
    # @TODO: Need to flesh this out :)
    
    # 2. make lst file
    
    lstname = LST_FILE_NAME
    
    status = 'NOT DONE'
    
    status = generate_lst_file(df, lstname)
    
    print('made lst file status: ', status)
    
    # 3. make rec file
    
    resize_val = 512
    lstlocation = 'tape-exp-test.lst'
    root_folder = r'C:\Users\Administrator\Desktop\notebooks-for-ml-ops\May15-tape-exp-data' #folder where images are stored locally
    rec_name = r'C:\Users\Administrator\Desktop\notebooks-for-ml-ops\tape-exp-test.rec'
    file_name = 'tape-exp-test.rec'
    s3_bucket = "tape-experiment-april6"
    generate_rec_file(resize_val,lstlocation, root_folder, rec_name, file_name, s3_bucket )
    
    # 4. invoke sagemaker, upload rec files to aws
    
    sagemakerRoleName = 'SkylerSageMakerRole'
    bucket = 'tubes-tape-exp-models'
    model_framework = "object-detection"
    prefix = 'retinanet'
    model_bucket_path = "s3://tubes-tape-exp-models/"
    s3_output_location = "s3://{}/{}/output".format(bucket,prefix)
    inst_type = "ml.p3.8xlarge" #<---costly, but fast. 21 mins per 100 epochs for 500img dataset
    region = 'us-east-1'
    
    sess = sagemaker.Session()

    try:
        role = sagemaker.get_execution_role()
    except ValueError:
        iam = boto3.client('iam')
        role = iam.get_role(RoleName=sagemakerRoleName)['Role']['Arn']
    
    training_image = image_uris.retrieve(
        region = sess.boto_region_name,
        framework = model_framework,
        version="1"
    )
    
    #print('Sagemaker session :', sess)
    print('default bucket :', bucket)
    print('Prefix :', prefix)
    print('Region selected :', region)
    print('IAM role :', role)
    print(training_image)
    
    # -----------############ Sagemaker Datachannel 
    
    s3_output_location = "s3://{}/{}/output".format(bucket,prefix)
    print('output location: ', s3_output_location)

    training_image = image_uris.retrieve(

        region = sess.boto_region_name, framework = "object-detection", version="1"
    )

    print(training_image)
    
    """
    upload binary rec file to AWS
    """
    pipeline_session = PipelineSession(default_bucket = model_bucket_path)

    # Upload the RecordIO files to train and validation channels

    train_channel = "train"
    validation_channel = "validation"

    sess.upload_data(path=rec_name, bucket=bucket, key_prefix=train_channel)
    sess.upload_data(path=rec_name, bucket=bucket, key_prefix=validation_channel)

    print('what is sess default bucket: ', sess.default_bucket())

    s3_train_data = f"s3://{s3_bucket}".format(bucket, train_channel)
    s3_validation_data = f"s3://{s3_bucket}".format(bucket, validation_channel)

    print(s3_train_data)
    print(s3_validation_data)
        
    train_data = sagemaker.inputs.TrainingInput(
        s3_train_data,
        distribution="FullyReplicated",
        content_type="application/x-recordio",
        s3_data_type="S3Prefix",
    )
    validation_data = sagemaker.inputs.TrainingInput(
        s3_validation_data,
        distribution="FullyReplicated",
        content_type="application/x-recordio",
        s3_data_type="S3Prefix",
    )
    
    print('data channels here!!')
    data_channels = {"train": train_data, "validation": validation_data}
    

    #data_channels = sagemaker_datachannels(rec_name,bucket,prefix,model_bucket_path,s3_bucket)
    #sagemaker_datachannels(rec_name,bucket,prefix,model_bucket_path,s3_bucket)
    
    # 5. define model

    #od_mdl = sagemaker_define_model(role, inst_type, training_image, s3_output_location)
    od_mdl = sagemaker.estimator.Estimator(
        training_image,
        role,
        instance_count = 1,
        instance_type = inst_type,
        volume_size = 50,
        max_run = 360000,
        input_mode = "File",
        output_path = s3_output_location,
        sagemaker_session = sess
    )
    
    num_classes = 4
    num_training_samples = 768
    num_epochs = 50
    lr_steps = '33,67'
    
    print('num classes: {}, num training images: {}'.format(num_classes, num_training_samples))

    od_mdl.set_hyperparameters(base_network='resnet-50',
                                 use_pretrained_model=1,
                                 num_classes=num_classes,
                                 mini_batch_size=64,
                                 epochs=num_epochs,               
                                 learning_rate=0.0002, 
                                 lr_scheduler_step=lr_steps,      
                                 lr_scheduler_factor=0.1,
                                 optimizer='adam',
                                 momentum=0.9,
                                 weight_decay=0.0005,
                                 overlap_threshold=0.5,
                                 nms_threshold=0.45,
                                 image_shape=512,
                                 label_width=350,
                                 num_training_samples=num_training_samples)
    
    # 6. kickoff training
    
    od_mdl.fit(inputs=data_channels, logs=True)
    
    #sagemaker_run_training(od_mdl,data_channels)
    print('model is kicked off in aws!')
    
    #notify_on_success(after=["sagemaker_run_training"])
    