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

from utils.im2rec import im2rec_entrypoint

# from steps import (
#     compute_performance_metrics_on_current_data,
#     dataloader_labelstudio,
#     hp_tuning_select_best_model,
#     hp_tuning_single_search,
#     model_evaluator,
#     model_trainer,
#     notify_on_failure,
#     notify_on_success,
#     promote_with_metric_compare,
#     train_data_preprocessor,
#     train_data_splitter,
# )

from steps import (
    compute_performance_metrics_on_current_data,
    dataloader_labelstudio,
    notify_on_failure
)


from zenml import pipeline
from zenml.logger import get_logger

logger = get_logger(__name__)

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


@pipeline(on_failure=notify_on_failure)
def e2e_use_case_training(
    model_search_space: Dict[str, Any],
    target_env: str,
    test_size: float = 0.2,
    drop_na: Optional[bool] = None,
    normalize: Optional[bool] = None,
    drop_columns: Optional[List[str]] = None,
    min_train_accuracy: float = 0.0,
    min_test_accuracy: float = 0.0,
    fail_on_accuracy_quality_gates: bool = False,
):
    """
    Model training pipeline.

    This is a pipeline that loads the data, processes it and splits
    it into train and test sets, then search for best hyperparameters,
    trains and evaluates a model.

    Args:
        model_search_space: Search space for hyperparameter tuning
        target_env: The environment to promote the model to
        test_size: Size of holdout set for training 0.0..1.0
        drop_na: If `True` NA values will be removed from dataset
        normalize: If `True` dataset will be normalized with MinMaxScaler
        drop_columns: List of columns to drop from dataset
        min_train_accuracy: Threshold to stop execution if train set accuracy is lower
        min_test_accuracy: Threshold to stop execution if test set accuracy is lower
        fail_on_accuracy_quality_gates: If `True` and `min_train_accuracy` or `min_test_accuracy`
            are not met - execution will be interrupted early
    """
    ### ADD YOUR OWN CODE HERE - THIS IS JUST AN EXAMPLE ###
    # Link all the steps together by calling them and passing the output
    # of one step as the input of the next step.
    ########## ETL stage ##########
    
    """
    Pull data from label studio environment. 
    Using dataloader_labelstudio takes 70+mins
    Using stored csv for now.
    """
    
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
    
    """
    prepare df for LST file
    """
    
    final = [] # for lst

    for i, row in df.iterrows():
        label = row['label']
        
        clss_lbl = one_hot_label(label)
        img_pth = row['local_filepath']
        s3_name = row['filename']
        img = cv2.imread(img_pth)
        h, w, c, = img.shape
        
        #print(h,w,c)
        
        #print(row['filename'])
        
        x_min_n = float(row['xmin']) / w
        x_max_n = float(row['xmax']) / w
        y_min_n = float(row['ymin']) / h
        y_max_n = float(row['ymax']) / h
        
        final.append([i, 2, 5, clss_lbl, x_min_n, y_min_n, x_max_n, y_max_n, s3_name])
        print(i, 2, 5, clss_lbl, x_min_n, y_min_n, x_max_n, y_max_n, s3_name)
    
    """
    write LST file
    """
    # now write out .lst file
    lstname = os.getcwd() + "\\"+"tape-exp-test.lst"
    with open(lstname, 'w', newline = '') as out:
        for row in final:
            writer = csv.writer(out, delimiter = '\t')
            writer.writerow(row)
            
    print('.lst is made here: ', lstname)
    
    """
    Make .rec file for AWS
    """
    
    # make rec file for AWS injection

    RESIZE_SIZE = 512 # 256 adjust this later # 512 gave us 0.74 mAP 
    
    #train_dir = r'C:\Users\Administrator\Desktop\april6-tape-exp-data'
    
    #folder where images are stored locally 
    train_dir = r'C:\Users\Administrator\Desktop\notebooks-for-ml-ops\May15-tape-exp-data'
    
    #parser.parse_args(['--sum', '7', '-1', '42'])
    
    # need to convert this into a call
    #!python tools/im2rec.py --resize $RESIZE_SIZE --pack-label tape-exp-test.lst $train_dir
    
    # invoke im2rec.py too
    
    resize_val = 512
        
    #im2rec_entrypoint(resize_val, lstname, train_dir)
    lstlocation = 'tape-exp-test.lst'
    root_folder = r'C:\Users\Administrator\Desktop\notebooks-for-ml-ops\May15-tape-exp-data'
    os.system(f"python utils\im2rec.py --resize {RESIZE_SIZE} --pack-label {lstlocation} {root_folder}")

    
    """
    send rec file to s3
    """
    
    rec_name = r'C:\Users\Administrator\Desktop\notebooks-for-ml-ops\tape-exp-test.rec'
    file_name = 'tape-exp-test.rec'
    s3_bucket = "tape-experiment-april6"
    s3 = boto3.resource('s3')

    # send image to s3
    s3.Bucket(s3_bucket).upload_file(rec_name, file_name)
    
    
    """
    Invoke Sagemaker to train model
    """

    import sagemaker
    from sagemaker import get_execution_role
    from sagemaker import image_uris

    bucket = 'tubes-tape-exp-models'
    prefix = 'retinanet'
    s3_output_location = "s3://{}/{}/output".format(bucket,prefix)

    print('output location: ', s3_output_location)

    sess = sagemaker.Session()
    training_image = image_uris.retrieve(
        
        region = sess.boto_region_name, framework = "object-detection", version="1"
    )

    print(training_image)
    
    """
    upload binary rec file to AWS
    """
    from sagemaker.workflow.pipeline_context import PipelineSession

    pipeline_session = PipelineSession(default_bucket = "s3://tubes-tape-exp-models/")

    # Upload the RecordIO files to train and validation channels

    train_channel = "train"
    validation_channel = "validation"

    sess.upload_data(path="tape-exp-test.rec", bucket=bucket, key_prefix=train_channel)
    sess.upload_data(path="tape-exp-test.rec", bucket=bucket, key_prefix=validation_channel)

    print('what is sess default bucket: ', sess.default_bucket())

    s3_train_data = "s3://tape-experiment-april6".format(bucket, train_channel)
    s3_validation_data = "s3://tape-experiment-april6".format(bucket, validation_channel)

    print(s3_train_data)
    print(s3_validation_data)
    
    """invoke sagemaker creds"""
    
    sess = sagemaker.Session()

    try:
        role = sagemaker.get_execution_role()
    except ValueError:
        iam = boto3.client('iam')
        role = iam.get_role(RoleName='SkylerSageMakerRole')['Role']['Arn']
    region = 'us-east-1'

    print('Sagemaker session :', sess)
    print('default bucket :', bucket)
    print('Prefix :', prefix)
    print('Region selected :', region)
    print('IAM role :', role)
        
    """
    define model 

    https://aws.amazon.com/sagemaker/pricing/

    inst_type
    used before: ml.p3.2xlarge
    changing to ml.p3.8xlarge <--- need to change service quotas
    """

    inst_type = "ml.p3.8xlarge" #<---costly, and haven't been given 100% usage from AWS yet...

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
    
    """set hyperparams"""
    
    def set_hyperparameters(num_epochs, lr_steps):
        num_classes = 4
        num_training_samples = 768
        print('num classes: {}, num training images: {}'.format(num_classes, num_training_samples))

        od_model.set_hyperparameters(base_network='resnet-50',
                                    use_pretrained_model=1,
                                    num_classes=num_classes,
                                    mini_batch_size=64,
                                    epochs=num_epochs,               
                                    learning_rate=0.0002, 
                                    lr_scheduler_step=lr_steps,      
                                    lr_scheduler_factor=0.1,
                                    optimizer='adam',
                                    momentum=0.9,
                                    weight_decay=0.0005,
                                    overlap_threshold=0.5,
                                    nms_threshold=0.45,
                                    image_shape=512,
                                    label_width=350,
                                    num_training_samples=num_training_samples)
        
    set_hyperparameters(100, '33,67')
    
    """
    create train/test channels
    """
    
    train_data = sagemaker.inputs.TrainingInput(
    s3_train_data,
    distribution="FullyReplicated",
    content_type="application/x-recordio",
    s3_data_type="S3Prefix",
    )
    validation_data = sagemaker.inputs.TrainingInput(
        s3_validation_data,
        distribution="FullyReplicated",
        content_type="application/x-recordio",
        s3_data_type="S3Prefix",
    )
    data_channels = {"train": train_data, "validation": validation_data}
    
    """
    initiate training
    """

    od_model.fit(inputs=data_channels, logs=True)

    # raw_data, target, _ = data_loader(random_state=random.randint(0, 100))
    # dataset_trn, dataset_tst = train_data_splitter(
    #     dataset=raw_data,
    #     test_size=test_size,
    # )
    # dataset_trn, dataset_tst, _ = train_data_preprocessor(
    #     dataset_trn=dataset_trn,
    #     dataset_tst=dataset_tst,
    #     drop_na=drop_na,
    #     normalize=normalize,
    #     drop_columns=drop_columns,
    # )
    
    
    ########## Hyperparameter tuning stage ##########
    after = []
    search_steps_prefix = "hp_tuning_search_"
    for config_name, model_search_configuration in model_search_space.items():
        step_name = f"{search_steps_prefix}{config_name}"
        hp_tuning_single_search(
            id=step_name,
            model_package=model_search_configuration["model_package"],
            model_class=model_search_configuration["model_class"],
            search_grid=model_search_configuration["search_grid"],
            dataset_trn=dataset_trn,
            dataset_tst=dataset_tst,
            target=target,
        )
        after.append(step_name)
    best_model = hp_tuning_select_best_model(step_names=after, after=after)

    ########## Training stage ##########
    model = model_trainer(
        dataset_trn=dataset_trn,
        model=best_model,
        target=target,
    )
    model_evaluator(
        model=model,
        dataset_trn=dataset_trn,
        dataset_tst=dataset_tst,
        min_train_accuracy=min_train_accuracy,
        min_test_accuracy=min_test_accuracy,
        fail_on_accuracy_quality_gates=fail_on_accuracy_quality_gates,
        target=target,
    )
    ########## Promotion stage ##########
    latest_metric, current_metric = (
        compute_performance_metrics_on_current_data(
            dataset_tst=dataset_tst,
            target_env=target_env,
            after=["model_evaluator"],
        )
    )

    promote_with_metric_compare(
        latest_metric=latest_metric,
        current_metric=current_metric,
        target_env=target_env,
    )
    last_step = "promote_with_metric_compare"

    notify_on_success(after=[last_step])
    ### YOUR CODE ENDS HERE ###
