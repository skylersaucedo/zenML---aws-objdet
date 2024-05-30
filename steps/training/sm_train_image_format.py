"""
use this to train with AWS Sagemaker MXNET Image format approach

ref: https://docs.aws.amazon.com/sagemaker/latest/dg/object-detection.html

"""

import pandas as pd
from typing import Any, Dict, List, Optional
import cv2
import boto3
import os
import csv
import argparse

from zenml import step
from zenml.io import fileio
from zenml.logger import get_logger
from typing_extensions import Annotated
    

from utils.current_labels import one_hot_json

categories = one_hot_json()

@step
def generate_sm_annos(
    df : pd.DataFrame) -> Annotated[str, "lst file made"] :
    
    ## generate SageMkaer Obj Det annotation json file

    for i, row in df.iterrows():
        label = row['label']
        
        #clss_lbl = one_hot_label(label)
        img_pth = row['local_filepath']
        s3_name = row['filename']
        img = cv2.imread(img_pth)
        h, w, c, = img.shape
                    
        x_min_n = float(row['xmin']) / w
        x_max_n = float(row['xmax']) / w
        y_min_n = float(row['ymin']) / h
        y_max_n = float(row['ymax']) / h
        
        anno = {
            'file' : img_pth,
            'image_size': [
                {'width' : w, 'height' : h, 'depth': c }
            ]
        }