# Zen ML and Computer Vision

Using ZenML as an MLOPs platform for AWS Object Detection and Sagemaker

## Description

This is a project that uses ZenML to manage and orchestrate an object detection Computer Vision Model that uses AWS Sagemaker. This project also uses Label studio to label videos. The goal is to use pipelines to continously train a model on new defects classes as new data is annotated and stored in S3 buckets.

## Getting Started

### Dependencies

* Make sure you have valid AWS credentials, IAM Roles, and know your way around the awscli. You will also need access to the ZenML platform. 

### Installing

* pip install -r requirements.txt

### Executing program

* conda create -n zenML python==3.11
* conda activate zenML
* pip install -r requirements.txt

Use Zenconnect command to authorize device

You will need to run the Makefile manually if your env is not Linux.
```
install-stack-local:
	@echo "Specify stack name [$(stack_name)]: " && read input && [ -n "$$input" ] && stack_name="$$input" || stack_name="$(stack_name)" && \
	zenml experiment-tracker register -f mlflow mlflow_local_$${stack_name} && \
	zenml model-registry register -f mlflow mlflow_local_$${stack_name} && \
	zenml model-deployer register -f mlflow mlflow_local_$${stack_name} && \
	zenml data-validator register -f evidently evidently_$${stack_name} && \
	zenml stack register -a default -o default -r mlflow_local_$${stack_name} \
	-d mlflow_local_$${stack_name} -e mlflow_local_$${stack_name} -dv \
	evidently_$${stack_name} $${stack_name} && \
	zenml stack set $${stack_name} && \
	zenml stack up
```

## Help

```
Please reach out to me if you have any questions. 
```

## Authors

e2e template provided by zenML: https://github.com/zenml-io/zenml/tree/main/examples/e2e

