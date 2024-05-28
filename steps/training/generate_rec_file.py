import os
from sklearn.model_selection import train_test_split
import argparse
from zenml import step
from zenml.io import fileio
from zenml.logger import get_logger
import boto3
from typing_extensions import Annotated
from typing import Any, Dict, List, Optional


logger = get_logger(__name__)

"""
Make .rec file for AWS
files must be locally downloaded, so EC2 instance 
may be needed
"""

@step
def generate_rec_file(
    resize_val :int,
    lstlocation : str,
    root_folder : str,
    rec_name : str,
    file_name : str, 
    s3_bucket : str) -> Annotated[str, "rec file made"]:

    # make rec file for AWS injection
    os.system(f"python utils\im2rec.py --resize {resize_val} --pack-label {lstlocation} {root_folder}") # run im2rec.py

    """
    send rec file to s3
    """
    s3 = boto3.resource('s3')

    # send image to s3
    s3.Bucket(s3_bucket).upload_file(rec_name, file_name)
    
    return "rec file made"