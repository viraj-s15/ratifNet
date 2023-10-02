import logging
from typing import Tuple

import pandas as pd
from sklearn.base import RegressorMixin
from typing_extensions import Annotated
from zenml import step

from src.evaluate import MSE, R2, RMSE


@step
def evaluate(
    model: RegressorMixin, X_test: pd.DataFrame, y_test: pd.DataFrame
) -> Tuple[Annotated[float, "r2_score"], Annotated[float, "rmse"]]:
    try:
        prediction = model.predict(X_test)
        mse_class = MSE()
        mse = mse_class.evaluate_score(y_test, prediction)
        r2 = R2()
        r2_score = r2.evaluate_score(y_test, prediction)
        rmse_class = RMSE()
        rmse = rmse_class.evaluate_score(y_test, prediction)
        return r2_score, rmse
    except Exception as e:
        logging.info(f"Error while evalutating model: {e}")
        raise e
