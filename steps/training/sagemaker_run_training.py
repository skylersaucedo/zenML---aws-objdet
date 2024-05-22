
import boto3

import sagemaker
from sagemaker import get_execution_role
from sagemaker import image_uris

import argparse
from zenml import step
from zenml.io import fileio
from zenml.logger import get_logger

logger = get_logger(__name__)

    
"""
initiate training on AWS Sagemaker Object Detector
"""

# async def run_model_fit(od_model,data_channels):
#     """use this so it doesn't muck up the pipeline"""
#     od_model.fit(inputs=data_channels, logs=True)
    
@step
def sagemaker_run_training(od_model,data_channels):
    od_model.fit(inputs=data_channels, logs=True)
    #run_model_fit(od_model,data_channels)
