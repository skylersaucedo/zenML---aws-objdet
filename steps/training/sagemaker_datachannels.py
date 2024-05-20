
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
def sagemaker_datachannels():
    
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
    