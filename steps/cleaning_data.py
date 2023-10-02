import logging
import os
from typing import Tuple

import pandas as pd
from typing_extensions import Annotated
from zenml import step

from src.data_cleaning import DataCleaning, DataPreprocessingStrategy, DataSplitStrategy


@step
def clean_data(
    df: pd.DataFrame,
) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.DataFrame, "y_train"],
    Annotated[pd.DataFrame, "y_train"],
]:
    try:
        process_strategy = DataPreprocessingStrategy()
        data_cleaning = DataCleaning(df, process_strategy)
        processed_data = data_cleaning.handle_data()

        split_strategy = DataCleaning(processed_data, DataSplitStrategy())
        X_train, X_test, y_train, y_test = data_cleaning.handle_data()
        logging.info("Data splitting is complete")
    except Exception as e:
        raise e
