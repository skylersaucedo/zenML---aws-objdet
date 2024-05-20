
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

logger = get_logger(__name__)

"""
Pull data from label studio environment. 
Using dataloader_labelstudio takes 70+mins
Using stored csv for now.

We will need to create a EC2 instance where images from videos are annoated to generate rec file...

returns train and test dataframes
"""

@step
def pull_annos_from_labelstudio():
    
    #df, target, random_state = dataloader_labelstudio("1")
    
    # use preloaded CSV file for df
    csv_file_name = os.getcwd() + "\\"+"may15annos.csv"
    df = pd.read_csv(csv_file_name)
    target = "target-csv" 
    random_state = 42
    
    """
    split df into test/train 
    """

    split = 0.8 # train/test split ratio
    train_df, test_df = train_test_split(df, test_size=1-split)
    
    return train_df, test_df 