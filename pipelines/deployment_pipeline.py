import json
import os

import numpy as np
import pandas as pd

# from materializer.custom_materializer import cs_materializer
from zenml import pipeline, step
from zenml.config import DockerSettings
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
    MLFlowModelDeployer,
)
from zenml.integrations.mlflow.services.mlflow_deployment import MLFlowDeploymentService
from zenml.integrations.mlflow.steps.mlflow_deployer import mlflow_model_deployer_step
from zenml.steps import BaseParameters, Output

from steps.cleaning_data import clean_data
from steps.evaluate import evaluate
from steps.ingest_data import ingest_data
from steps.train import train_model

# from .utils import get_data_for_test

docker_settings = DockerSettings(required_integrations=[MLFLOW])


class DeploymentTriggerConfig(BaseParameters):
    # min_accuracy: float = 0.92
    min_accuracy: float = 0


@step
def deployment_trigger(accuracy: float, config: DeploymentTriggerConfig):
    return accuracy >= config.min_accuracy


@pipeline(enable_cache=False, settings={"docker": docker_settings})
def continuous_deployment_pipeline(
    data_path: str,
    min_accuracy: float = 0,
    workers: int = 1,
    timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT,
):
    df = ingest_data(data_path)
    X_train, X_test, y_train, y_test = clean_data(df)
    model = train_model(X_train, X_test, y_train, y_test)
    r2_score, rmse = evaluate(model, X_test, y_test)
    deployment_decision = deployment_trigger(r2_score)
    mlflow_model_deployer_step(
        model=model,
        deploy_decision=deployment_decision,
        workers=workers,
        timeout=timeout,
    )
