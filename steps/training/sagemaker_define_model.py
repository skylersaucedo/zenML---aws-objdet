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
define model 

https://aws.amazon.com/sagemaker/pricing/

inst_type
used before: ml.p3.2xlarge
changing to ml.p3.8xlarge <--- need to change service quotas
"""

@step
def make_od_model():
    inst_type = "ml.p3.8xlarge" #<---costly, but fast. 21 mins per 100 epochs for 500img dataset

    od_model = sagemaker.estimator.Estimator(
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

    print(od_model)