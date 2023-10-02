import logging

import pandas as pd
from sklearn.base import RegressorMixin
from zenml import step

from src.model_dev import LinearRegressionModel

from .config import ModelNameConfig


@step
def train_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    config: ModelNameConfig,
):
    try:
        model = None
        if config.model_name == "LinearRegression":
            model = LinearRegressionModel()
            trained_model = model.train(X_train, y_train)
            return trained_model
    except Exception as e:
        logging.info("Something went wrong in the model training")
        raise e
