
import random
from typing import Any, Dict, List, Optional
import cv2
import boto3
import pandas as pd
import os
import csv
from sklearn.model_selection import train_test_split
import argparse
from zenml import step
from zenml.io import fileio
from zenml.logger import get_logger
from typing_extensions import Annotated

from steps.etl.dataloader_labelstudio import dataloader_labelstudio

logger = get_logger(__name__)

"""
Pull data from label studio environment. 
Using dataloader_labelstudio takes 70+mins
Using stored csv for now.

We will need to create a EC2 instance where images from videos are annoated to generate rec file...

returns train and test dataframes
"""

@step
def pull_annos_from_labelstudio(
    iscsv : str, 
    csv_file_name : str) -> pd.DataFrame:
    
    if iscsv:
        # use preloaded CSV file for df
        df = pd.read_csv(csv_file_name)
        target = "target-csv" 
        random_state = 42

    #split = 0.8 # train/test split ratio
    #train_df, test_df = train_test_split(df, test_size=1-split)
    
    else:
        df, target, random_state = dataloader_labelstudio("1")
        
        logger.info(f"pulled new data from: {target}")
        logger.info(df.head())
        
    return df