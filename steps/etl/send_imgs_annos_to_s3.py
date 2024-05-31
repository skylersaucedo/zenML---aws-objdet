
import json
import boto3
import matplotlib.pyplot as plt
import cv2
from utils.current_labels import one_hot_json, one_hot_label
import os
from zenml import step
from zenml.logger import get_logger
import pandas as pd

s3 = boto3.resource('s3')

images_bucket = 'tape-exp-images-may30'
annos_bucket = 'tape-exp-annos-may30'

logger = get_logger(__name__)

@step
def send_data_to_s3(
    df : pd.DataFrame,
    images_bucket : str, 
    annos_bucket : str) -> str:
    """send images and annos to s3 buckets"""

    for i, row in df.iterrows():
            label = row['label']
            
            clss_lbl = one_hot_label(label)
            img_pth = row['local_filepath']
            s3_name = row['filename']
            
            img_r = cv2.imread(img_pth)
            img = cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB) # convert to correct color sequence    

            h, w, c, = img.shape
            
            x_min = float(row['xmin'])
            x_max = float(row['xmax'])
            y_min = float(row['ymin'])
            y_max = float(row['ymax'])
            
            left = x_min
            top = y_max
            width = x_max - x_min
            height = y_max - y_min
            
            #s3_img_path = 'mybucket' + img_pth
            
            filename = os.path.basename(img_pth)
            
            logger.info('filename: ', filename)
            
            s = filename.split('.')
            
            anno = {
                'file' : filename,
                'image_size': [
                    {'width' : w, 'height' : h, 'depth': c }
                ],
                'categories' : one_hot_json(),
                
                'annotations' : [
                    {
                        "class_id": clss_lbl,
                        "left": left,
                        "top": top,
                        "width": width,
                        "height": height
                        }
                    ]
            }
            
            logger.info(anno)
                    
            #### ----- PLOT HERE TO VERIFY ---- ######
            # start_point = (int(left),int(top)) # top left
            # end_point = (int(left+width),int(y_min)) # bottom right
            #         # add rect
            # color = (0, 0, 255) # Blue color in RGB
            # thickness = 10 # Line thickness
            # image = cv2.rectangle(img, start_point, end_point, color, thickness)
            # plt.figure(figsize=(3,3))
            # plt.imshow(image)
            # plt.title(label)
            # plt.show()

            # save anno to s3
            
            anno_filename = s[0] +'.json'
            s3JSONobject = s3.Object(annos_bucket, anno_filename)
            s3JSONobject.put(
                Body=(bytes(json.dumps(anno).encode('UTF-8')))
                )
                    
            #send image to s3
            s3.Bucket(images_bucket).upload_file(img_pth, filename)
            logger.info('file saved to s3!', filename)
            
    return "sent stuff to s3!"
                    