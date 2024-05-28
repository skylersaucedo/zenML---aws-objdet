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

logger = get_logger(__name__)


"""
prepare df for LST file
"""

def one_hot_label(label):
    """
    one hot encode label from string to int
    """
    r = 4
    if label == 'blue_tape':
        r = 0
    if label == 'black_tape':
        r = 1
    if label == 'gum':
        r = 2
    if label == 'leaf':
        r = 3
        
    return r

@step
def generate_lst_file(
    df : pd.DataFrame, 
    lstname: str) -> Annotated[str, "lst file made"] :

    final = [] # for lst

    for i, row in df.iterrows():
        label = row['label']
        
        clss_lbl = one_hot_label(label)
        img_pth = row['local_filepath']
        s3_name = row['filename']
        img = cv2.imread(img_pth)
        h, w, c, = img.shape
                    
        x_min_n = float(row['xmin']) / w
        x_max_n = float(row['xmax']) / w
        y_min_n = float(row['ymin']) / h
        y_max_n = float(row['ymax']) / h
        
        final.append([i, 2, 5, clss_lbl, x_min_n, y_min_n, x_max_n, y_max_n, s3_name])
        logger.info(f"lst input: {i, 2, 5, clss_lbl, x_min_n, y_min_n, x_max_n, y_max_n, s3_name}")

    """
    write LST file
    """
    # now write out .lst file
    
    logger.info("writing LST file..")
    
    with open(lstname, 'w', newline = '') as out:
        for row in final:
            writer = csv.writer(out, delimiter = '\t')
            writer.writerow(row)
            
    return "lst file made"