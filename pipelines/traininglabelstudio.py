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

from steps import (
    pull_annos_from_labelstudio,
    send_data_to_s3,
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
    num_classes: int,
    num_training_samples: int,
    num_epochs: int,
    lr_steps: str,
    base_network: str,   
    mini_batch_size: int,
    lr: float,
    lrsf: float,
    opt: str,
    momentum: float,
    weight_decay: float,
    overlap_threshold: float,
    nms_threshold: float,
    image_shape: int,
    label_width: int
):
    """
    Model training pipeline.

    This is a pipeline - flesh out description later...

    Args:

    """
    #train_dataset_id: Optional[UUID] = None,
    #test_dataset_id: Optional[UUID] = None,

    # 1. grab annos from label studio
    
    csv_file_name = CSV_FILE_NAME
    iscsv = True # use this before directly connecting label studio

    df = pull_annos_from_labelstudio(iscsv,csv_file_name)
    
    print(df)
    
    # 2. Send images and JSON annotations to AWS S3
    
    status = send_data_to_s3(df)
    
    ### @TODO - Check to see if image already exists in bucket!
    
    print('data sent to s3? ', status)
    
    # 3. define image and anno channels
    
    ## images and jsons are in S3 buckets, 
    # test channels to see if training spools up
    images_bucket = 'tape-exp-images-may30'
    annos_bucket = 'tape-exp-annos-may30'

    # same buckets for now, change later!
    s3_train_images = 's3://{}/'.format(images_bucket)
    s3_val_images = 's3://{}/'.format(images_bucket)
    s3_train_anno = 's3://{}/'.format(annos_bucket)
    s3_val_anno = 's3://{}/'.format(annos_bucket)
    
    # 4. invoke sagemaker, spool up session
    
    sagemakerRoleName = 'SkylerSageMakerRole'
    bucket = 'tubes-tape-exp-models'
    model_framework = "object-detection"
    prefix = 'retinanet'
    model_bucket_path = "s3://tubes-tape-exp-models/"
    s3_output_location = "s3://{}/{}/output".format(bucket,prefix)
    inst_type = "ml.p3.8xlarge" #<---costly, but fast. 21 mins per 100 epochs for 500img dataset
    region = 'us-east-1'
    
    sess = sagemaker.Session()
    print('Sagemaker session :', sess)

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
    
    print('default bucket :', bucket)
    print('Prefix :', prefix)
    print('Region selected :', region)
    print('IAM role :', role)
    print(training_image)
    
    # 5. Spool up Sagemaker Pipeline training object
    model_bucket_path = "s3://tubes-tape-exp-models/"
    training_image = image_uris.retrieve(

        region = sess.boto_region_name, framework = "object-detection", version="1"
    )

    print(training_image)

    """
    define pipeline session and buckets 
    """
    
    pipeline_session = PipelineSession(default_bucket = model_bucket_path)
    print(pipeline_session)
    
    ### ----- define sagemaker training inputs
        
    train_data_images = sagemaker.inputs.TrainingInput(
        s3_train_images,
        distribution="FullyReplicated",
        content_type="application/x-image",
        s3_data_type="S3Prefix",
    )
    validation_data_images = sagemaker.inputs.TrainingInput(
        s3_val_images,
        distribution="FullyReplicated",
        content_type="application/x-image",
        s3_data_type="S3Prefix",
    )

    train_data_annotations = sagemaker.inputs.TrainingInput(
        s3_train_anno,
        distribution="FullyReplicated",
        content_type="application/x-image",
        s3_data_type="S3Prefix",
    )
    validation_data_annotations = sagemaker.inputs.TrainingInput(
        s3_val_anno,
        distribution="FullyReplicated",
        content_type="application/x-image",
        s3_data_type="S3Prefix",
    )

    print('data channels here!!')
    data_channels = {
        "train": train_data_images, 
        "validation": validation_data_images,
        "train_annotation" : train_data_annotations,
        "validation_annotation" : validation_data_annotations
        }
    
    print('num classes: {}, num training images: {}'.format(num_classes, num_training_samples))
    
    # 6. kickoff training

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
    
    od_mdl.set_hyperparameters(base_network=base_network,
                                 use_pretrained_model=1,
                                 num_classes=num_classes,
                                 mini_batch_size=mini_batch_size,
                                 epochs=num_epochs,               
                                 learning_rate=lr, 
                                 lr_scheduler_step=lr_steps,      
                                 lr_scheduler_factor=lrsf,
                                 optimizer=opt,
                                 momentum=momentum,
                                 weight_decay=weight_decay,
                                 overlap_threshold=overlap_threshold,
                                 nms_threshold=nms_threshold,
                                 image_shape=image_shape,
                                 label_width=label_width,
                                 num_training_samples=num_training_samples)
        
    od_mdl.fit(inputs=data_channels, logs=True)
    
    print('model is kicked off in aws!')
    
    #notify_on_success(after=["sagemaker_run_training"])
    
    if __name__ == "__main__":
        training_pipeline()
    