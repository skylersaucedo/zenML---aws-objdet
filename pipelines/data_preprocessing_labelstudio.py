"""
Use this to pull data from label studio
raw data is annotated videos
we need to convert to images and save to S3
Make sure AWS credentials are stored in .env 
"""

from label_studio_sdk import Client
import dotenv
import os
import json
import pandas as pd
import numpy as np
import boto3
import matplotlib.pyplot as plt
from datetime import datetime
import cv2

dotenv.load_dotenv()

API_ROOT = os.getenv("LS_API_ROOT")
LS_API_TOKEN = os.getenv("LS_API_TOKEN")

# Initialize the Label Studio SDK client with the additional headers
ls = Client(url=API_ROOT, api_key=LS_API_TOKEN)

id = "1"
exportType = "JSON"
output_filename = 'may14annotations.json'
csv_path = os.getcwd() + "\\"+"may15annos.csv"

# image must be saved locally to geenrate binary from img2rec.py
# 

def export_labelstudio_video_annos_csv(ls,id,csv_path):
    
    # csv
    project = ls.get_project(id)
    project.export_tasks(export_type='CSV', export_location=csv_path)

def export_labelstudio_video_annotations(ls,id):
    """generate tasks JSON to create dataset"""
    # Get the project
    project = ls.get_project(id)

    # Export tasks
    tasks = project.export_tasks(export_type='JSON', download_all_tasks=True)
    return tasks

def create_dataset(tasks):
    """
    pass label studio tasks JSON object, return dataframe 
    task is JSON video label
    dataframe will have: 
    'local_filepath','filename','label','xmin','xmax','ymin','ymax','pipe_id','dt','side','passnum','cam','video_name','frame'
    ### images saved locally and to AWS s3 bucket
    """
    annotations = []
    img_data_savepath =  os.getcwd() + "\\"+ "May15-tape-exp-data"
    s3_bucket_savedimages = "tape-experiment-april6"
    
    for i, job in enumerate(tasks): # each task is a labeled video
        
        annos = job['annotations']
        data = job['data']
            
        if len(annos) == 1: # video is annotated
                    
            vid_s3_path = data['video']
            print('viewing annos for: ', vid_s3_path)
            
            # @TODO: flesh out info from vid name...
            pipe_id = '1000'
            side = 'BOX'
            passnum = '1'
            cam = '0'
            
            r = annos[0] # annos stored in list, so remove list              
            n = r['id'] # number of defects persistent during video
                    
            res = r['result']
            
            for j, defect in enumerate(res): # each defect persists by frame
                
                v = defect['value']
                label = v['labels'][0].replace(' ','_')            
                fc = v['framesCount']
                duration = v['duration']
                sequence = v['sequence']
    
                for k, obj in enumerate(sequence):
                                    
                    frame = obj["frame"]
                    x = obj["x"]
                    y = obj["y"]
                    width = obj["width"]
                    height = obj["height"]
                    
                    #load video from S3
                    s3 = boto3.client("s3")
                    bucket_name = "tsi-inc"
                    video_key = vid_s3_path.split('/')[-1]
                    
                    url = s3.generate_presigned_url( ClientMethod='get_object', Params={ 'Bucket': bucket_name, 'Key': video_key } )
                    
                    video = cv2.VideoCapture(url)
                    video.set(cv2.CAP_PROP_POS_FRAMES, frame)
                    ret, f_p = video.read()
                                    
                    if len(f_p) > 0: # need to make sure image is made from video
                        
                        f = cv2.cvtColor(f_p, cv2.COLOR_BGR2RGB) # convert to correct color sequence    
                        img = np.asarray(f)
                        
                        h, w, c = img.shape
                    
                        #resize bounding box locations.. they are in percentage
                        x = (x/100)*w
                        y = (y/100)*h
                                            
                        fudge = 35 # need to confirm this                    
                        height = fudge*height
                        width = fudge*width                
                        
                        # see image with bounding box
                        
                        # add rect
                        color = (0, 0, 255) # Blue color in RGB
                        thickness = 10 # Line thickness 
                        
                        start_point = (int(x),int(y+height)) # top left
                        end_point = (int(x+width),int(y)) # bottom right
                        
                        image = cv2.rectangle(img, start_point, end_point, color, thickness)
                        
                        xmin = int(x)
                        xmax = int(x + width)
                        ymin =int(y)
                        ymax = int(y + height)
                        
                        # use to plot for verificiation, its good...
                        
                        # plt.figure(figsize=(3,3))
                        # plt.imshow(image)
                        # plt.title(label)
                        # plt.show()
                        
                        # save locally, send to S3
                        
                        dt = datetime.today().strftime('%Y%m%d%H%M%S')
                        print('datetime: ', dt)
                        
                        filename = f"{pipe_id}_{dt}_{side}_{passnum}_{cam}_{frame}_{label}.png"
                        local_filepath = img_data_savepath + '\\' + filename
                        cv2.imwrite(local_filepath, f_p) # save image locally

                        # send image to s3
                        s3b = boto3.resource('s3')
                        s3b.Bucket(s3_bucket_savedimages).upload_file(local_filepath, filename)
                        print('file saved to s3!', local_filepath)
                        
                        # add image, annotation, label, to dataframe 
                        annotations.append([local_filepath,filename,label,xmin,xmax,ymin,ymax,pipe_id,dt,side,passnum,cam,filename,frame])

                        
                    else:
                        print('ERROR  with frame for: ', vid_s3_path)
                    
    # make df
    cols = ['local_filepath','filename','label','xmin','xmax','ymin','ymax','pipe_id','dt','side','passnum','cam','video_name','frame']
    return pd.DataFrame(data=annotations, columns=cols)



