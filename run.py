import uuid 
import os
from datetime import datetime as dt
from typing import Optional
import click

from pipelines import (
    training_pipeline
    
)

from utils.constants import(
    ZENML_MODEL_NAME,
    TRAINING_CONFIG_PATH
)

from zenml.logger import get_logger
from zenml.client import Client
#from zenml.enums import ModelStages

logger = get_logger(__name__)


@click.command(
    help="""
TSI ML OPS project CLI v1.0.0.

Run the ZenML E2E project model training pipeline with various
options.

Examples:


  \b
  # Run the pipeline with default options
  python run.py
               
  \b
  # Run the pipeline without cache
  python run.py --no-cache

  \b
  # Run the pipeline without Hyperparameter tuning
  python run.py --no-hp-tuning

  \b
  # Run the pipeline without NA drop and normalization, 
  # but dropping columns [A,B,C] and keeping 10% of dataset 
  # as test set.
  python run.py --no-drop-na --no-normalize --drop-columns A,B,C --test-size 0.1

  \b
  # Run the pipeline with Quality Gate for accuracy set at 90% for train set 
  # and 85% for test set. If any of accuracies will be lower - pipeline will fail.
  python run.py --min-train-accuracy 0.9 --min-test-accuracy 0.85 --fail-on-accuracy-quality-gates


"""
)

@click.option(
    "--tsi-train", 
    "--training",
    "-t",
    "train",
    is_flag = True,
    default = False,
    help="Activate TSI training sequence"
)

def main(
    train: bool = False
):
    """Main entry point for the pipeline execution.

    This entrypoint is where everything comes together:

      * configuring pipeline with the required parameters
        (some of which may come from command line arguments)
      * launching the pipeline

    Args:
        Add args here
    """

    # Run a pipeline with the required parameters. This executes
    # all steps in the pipeline in the correct order using the orchestrator
    # stack component that is configured in your active ZenML stack.
    
    client = Client()
    
    # Execute Training Pipeline
    
    if train:
        
        if not client.active_stack.orchestrator.config.is_local:
            raise RuntimeError(
                "The implementation of this pipeline "
                "requires that you are running on a local "
                "machine with data being persisted in the local "
                "filesystem across multiple steps. Please "
                "switch to a stack that contains a local "
                "orchestrator and a connected label-studio "
                "annotator. See the README for more information "
                "on this setup."
            )
            
        train_batch_size = 16
        eval_batch_size = 8
        learning_rate = 1e-4
        weight_decay = 1e-6
        img_size = 512
        dataset_artifact_id = str(uuid.uuid4())
        model_artifact_id = str(uuid.uuid4())
        num_classes = 4
        num_training_samples = 768
        num_epochs = 50
        lr_steps = '33,67'
        base_network ='resnet-50'   
        mini_batch_size = 64
        lr = 0.0002
        lrsf = 0.1
        opt = 'adam'
        momentum = 0.9
        weight_decay = 0.0005
        overlap_threshold=0.5
        nms_threshold=0.45
        image_shape=512
        label_width=350
        
        training_args = {
            "num_classes": num_classes,
            "num_training_samples":num_training_samples,
            "num_epochs" : num_epochs,
            "lr_steps" : lr_steps,
            "base_network" : base_network,   
            "mini_batch_size" : mini_batch_size,
            "lr" : lr,
            "lrsf" : lrsf,
            "opt" :opt,
            "momentum" : momentum,
            "weight_decay" : weight_decay,
            "overlap_threshold" : overlap_threshold,
            "nms_threshold": nms_threshold,
            "image_shape" : image_shape,
            "label_width" : label_width
        }
        
        # pipeline_args = {"config_path" : TRAINING_CONFIG_PATH,
        #                  "images_bucket" : 's3://{}/'.format('tape-exp-images-may30'),
        #                  "annos_bucket" : 's3://{}/'.format('tape-exp-annos-may30')
        #                  }
        
        pipeline_args = {"config_path" : TRAINING_CONFIG_PATH,
                    }

    
        pipeline_args["config_path"] = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "configs",
            "train_config.yaml",
        )
        pipeline_args["run_name"] = (
            f"tsimlopsdti_run_{dt.now().strftime('%Y_%m_%d_%H_%M_%S')}"
        )
        
        logger.info('type for pipeline_args: ', type(pipeline_args))
        logger.info('type for training_args: ', type(training_args))

        logger.info('pipeline args: ', pipeline_args)
        logger.info('run train agrs: ', training_args)
        
        training_pipeline.with_options(**pipeline_args)(**training_args)
   
        logger.info("Training pipeline finished successfully!")

if __name__ == "__main__":
    main()
