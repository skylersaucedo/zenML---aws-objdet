import uuid 
import os
from datetime import datetime as dt
from typing import Optional
import click

# from pipelines import (
#     e2e_use_case_batch_inference,
#     e2e_use_case_deployment,
#     e2e_use_case_training,
# )

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

# @click.option(
#     "--no-cache",
#     is_flag=True,
#     default=False,
#     help="Disable caching for the pipeline run.",
# )
# @click.option(
#     "--no-drop-na",
#     is_flag=True,
#     default=False,
#     help="Whether to skip dropping rows with missing values in the dataset.",
# )
# @click.option(
#     "--no-normalize",
#     is_flag=True,
#     default=False,
#     help="Whether to skip normalization in the dataset.",
# )
# @click.option(
#     "--drop-columns",
#     default=None,
#     type=click.STRING,
#     help="Comma-separated list of columns to drop from the dataset.",
# )
# @click.option(
#     "--test-size",
#     default=0.2,
#     type=click.FloatRange(0.0, 1.0),
#     help="Proportion of the dataset to include in the test split.",
# )
# @click.option(
#     "--min-train-accuracy",
#     default=0.8,
#     type=click.FloatRange(0.0, 1.0),
#     help="Minimum training accuracy to pass to the model evaluator.",
# )
# @click.option(
#     "--min-test-accuracy",
#     default=0.8,
#     type=click.FloatRange(0.0, 1.0),
#     help="Minimum test accuracy to pass to the model evaluator.",
# )
# @click.option(
#     "--fail-on-accuracy-quality-gates",
#     is_flag=True,
#     default=False,
#     help="Whether to fail the pipeline run if the model evaluation step "
#     "finds that the model is not accurate enough.",
# )
# @click.option(
#     "--only-inference",
#     is_flag=True,
#     default=False,
#     help="Whether to run only inference pipeline.",
#)
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
    
    # pipeline_args = {}
    # if no_cache:
    #     pipeline_args["enable_cache"] = False
        
    # Execute Training Pipeline
    
    if train:
        # try:
        #     client.get_model_version(
        #         model_name_or_id=ZENML_MODEL_NAME,
        #         #model_version_name_or_number_or_id=ModelStages.STAGING,
        #     )
        # except KeyError:
        #     raise RuntimeError(
        #         "This pipeline requires that there is a version of its "
        #         "associated model in the `STAGING` stage. Make sure you run "
        #         "the `data_export_pipeline` at least once to create the Model "
        #         "along with a version of this model. After this you can "
        #         "promote the version of your choice, either through the "
        #         "frontend or with the following command: "
        #         f"`zenml model version update {ZENML_MODEL_NAME} latest "
        #         f"-s staging`"
        #     )

        # if train_local:
        #     config_path = "configs/training_pipeline.yaml"
        # else:
        #     config_path = "configs/training_pipeline_remote_gpu.yaml"
            
        #config_path = "configs/training_pipeline.yaml"
        
        # training_pipeline.with_options(
        #     config_path="configs/train_config.yaml"
        # )()

        # Train model on data
        #training_pipeline.with_options(config_path=config_path)()
        #training_pipeline.with_options(config_path=config_path)

        #run_args_train = {}
        
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
            
        pipeline_args = {"config_path" : TRAINING_CONFIG_PATH}
        
        
        num_epochs = 20
        train_batch_size = 16
        eval_batch_size = 8
        learning_rate = 1e-4
        weight_decay = 1e-6
        img_size = 512
        dataset_artifact_id = str(uuid.uuid4())
        model_artifact_id = str(uuid.uuid4())
        
        # run_args_train = {
        #     "num_epochs": num_epochs,
        #     "train_batch_size": train_batch_size,
        #     "eval_batch_size": eval_batch_size,
        #     "learning_rate": learning_rate,
        #     "weight_decay": weight_decay,
        #     "img_size": img_size,
        #     "dataset_artifact_id": dataset_artifact_id,
        #     "model_artifact_id": model_artifact_id,
        # }
        
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
    
    # if not only_inference:
    #     # Execute Training Pipeline
    #     run_args_train = {
    #         "drop_na": not no_drop_na,
    #         "normalize": not no_normalize,
    #         "test_size": test_size,
    #         "min_train_accuracy": min_train_accuracy,
    #         "min_test_accuracy": min_test_accuracy,
    #         "fail_on_accuracy_quality_gates": fail_on_accuracy_quality_gates,
    #     }
    #     if drop_columns:
    #         run_args_train["drop_columns"] = drop_columns.split(",")

        pipeline_args["config_path"] = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "configs",
            "train_config.yaml",
        )
        pipeline_args["run_name"] = (
            f"tsimlopsdti_run_{dt.now().strftime('%Y_%m_%d_%H_%M_%S')}"
        )
        
        print('type for pipeline_args: ', type(pipeline_args))
        print('type for training_args: ', type(training_args))

        print('pipeline args: ', pipeline_args)
        print('run train agrs: ', training_args)
        
        #traininglabelstudio.with_options(**pipeline_args)(**run_args_train)
        #training_pipeline.with_options(**pipeline_args)(**run_args_train)
        training_pipeline.with_options(**pipeline_args)(**training_args)
        #training_pipeline.with_options()()
        #training_pipeline()

        logger.info("Training pipeline finished successfully!")

    # # Execute Deployment Pipeline
    # run_args_inference = {}
    # pipeline_args["config_path"] = os.path.join(
    #     os.path.dirname(os.path.realpath(__file__)),
    #     "configs",
    #     "deployer_config.yaml",
    # )
    # pipeline_args["run_name"] = (
    #     f"e2e_use_case_deployment_run_{dt.now().strftime('%Y_%m_%d_%H_%M_%S')}"
    # )
    # e2e_use_case_deployment.with_options(**pipeline_args)(**run_args_inference)

    # # Execute Batch Inference Pipeline
    # run_args_inference = {}
    # pipeline_args["config_path"] = os.path.join(
    #     os.path.dirname(os.path.realpath(__file__)),
    #     "configs",
    #     "inference_config.yaml",
    # )
    # pipeline_args["run_name"] = (
    #     f"e2e_use_case_batch_inference_run_{dt.now().strftime('%Y_%m_%d_%H_%M_%S')}"
    # )
    # e2e_use_case_batch_inference.with_options(**pipeline_args)(
    #     **run_args_inference
    # )


if __name__ == "__main__":
    main()
