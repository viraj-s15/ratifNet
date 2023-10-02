import logging
from typing import Tuple
import mlflow
import pandas as pd
from sklearn.base import RegressorMixin
from typing_extensions import Annotated
from zenml import step
from zenml.client import Client
from src.evaluate import MSE, R2, RMSE

experment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experment_tracker.name)
def evaluate(
    model: RegressorMixin, X_test: pd.DataFrame, y_test: pd.DataFrame
) -> Tuple[Annotated[float, "r2_score"], Annotated[float, "rmse"]]:
    try:
        prediction = model.predict(X_test)
        mse_class = MSE()
        mse = mse_class.evaluate_score(y_test, prediction)
        mlflow.log_metric("mse",mse)
        r2_class = R2()
        r2 = r2_class.evaluate_score(y_test, prediction)
        mlflow.log_metric("r2",r2)
        rmse_class = RMSE()
        rmse = rmse_class.evaluate_score(y_test, prediction)
        mlflow.log_metric("rmse",rmse)
        return r2, rmse
    except Exception as e:
        logging.info(f"Error while evalutating model: {e}")
        raise e
