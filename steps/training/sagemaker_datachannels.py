
from zenml import step
from zenml.io import fileio
from zenml.logger import get_logger
import sagemaker
from sagemaker import get_execution_role
from sagemaker import image_uris

from sagemaker.workflow.pipeline_context import PipelineSession

logger = get_logger(__name__)


"""
Invoke Sagemaker to train model
"""

@step
def sagemaker_datachannels(rec_name,bucket,prefix,model_bucket_path,s3_bucket):
    print('data channel goes here...')
    # sess = sagemaker.Session()
    # s3_output_location = "s3://{}/{}/output".format(bucket,prefix)
    # print('output location: ', s3_output_location)

    # training_image = image_uris.retrieve(

    #     region = sess.boto_region_name, framework = "object-detection", version="1"
    # )

    # print(training_image)
    
    # """
    # upload binary rec file to AWS
    # """
    # pipeline_session = PipelineSession(default_bucket = model_bucket_path)

    # # Upload the RecordIO files to train and validation channels

    # train_channel = "train"
    # validation_channel = "validation"

    # sess.upload_data(path=rec_name, bucket=bucket, key_prefix=train_channel)
    # sess.upload_data(path=rec_name, bucket=bucket, key_prefix=validation_channel)

    # print('what is sess default bucket: ', sess.default_bucket())

    # s3_train_data = f"s3://{s3_bucket}".format(bucket, train_channel)
    # s3_validation_data = f"s3://{s3_bucket}".format(bucket, validation_channel)

    # print(s3_train_data)
    # print(s3_validation_data)
        
    # train_data = sagemaker.inputs.TrainingInput(
    #     s3_train_data,
    #     distribution="FullyReplicated",
    #     content_type="application/x-recordio",
    #     s3_data_type="S3Prefix",
    # )
    # validation_data = sagemaker.inputs.TrainingInput(
    #     s3_validation_data,
    #     distribution="FullyReplicated",
    #     content_type="application/x-recordio",
    #     s3_data_type="S3Prefix",
    # )
    # data_channels = {"train": train_data, "validation": validation_data}
    
    # return training_image, data_channels
    