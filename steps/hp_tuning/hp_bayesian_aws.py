"""
AWS RetinaNet Object Detection
"""

"""run bayesian hyperparameter search in Sagemaker"""
import sagemaker
import boto3
from sagemaker import image_uris


### Sagemaker roles

sess   = sagemaker.Session()
bucket = sess.default_bucket()                     
prefix = 'objectdetection'
#region = boto3.Session().region_name
region = 'us-east-1'

try:
    role = sagemaker.get_execution_role()
except ValueError:
    iam = boto3.client('iam')
    role = iam.get_role(RoleName='SkylerSageMakerRole')['Role']['Arn']

print('Sagemaker session :', sess)
print('default bucket :', bucket)
print('Prefix :', prefix)
print('Region selected :', region)
print('IAM role :', role)

"""
use this to train your model
"""


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
Add datapipeline config
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

"""
https://docs.aws.amazon.com/sagemaker/latest/dg/object-detection-api-config.html
"""

def set_hyperparameters(num_epochs, lr_steps):
    num_classes = 4
    num_training_samples = 760
    print('num classes: {}, num training images: {}'.format(num_classes, num_training_samples))

    od_model.set_hyperparameters(base_network='resnet-50',
                                 use_pretrained_model=1,
                                 num_classes=num_classes,
                                 mini_batch_size=16,
                                 epochs=num_epochs,               
                                 learning_rate=0.0001, 
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
    
        
set_hyperparameters(100, "50,70,80,90,95")

"""
hyperparameter tuning
AWS does Bayesian search by default.
"""

from sagemaker.tuner import CategoricalParameter, ContinuousParameter, HyperparameterTuner


hyperparameter_ranges = {
    
    "learning_rate": ContinuousParameter(0.0001,0.0003),
    "mini_batch_size": CategoricalParameter([16,32,64,128]),
    "optimizer": CategoricalParameter(["adadelta","adam"])

}


max_jobs = 16
max_parallel_jobs = 2
objective_metric_name = "validation:mAP"
objective_type = "Maximize"

tuner = HyperparameterTuner(estimator = od_model,
                            objective_metric_name = objective_metric_name,
                            hyperparameter_ranges = hyperparameter_ranges,
                            objective_type = objective_type,
                            max_jobs = max_jobs,
                            max_parallel_jobs = max_parallel_jobs          
                           )

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

print(data_channels)

# make it so.

tuner.fit(inputs = data_channels, logs = True)
