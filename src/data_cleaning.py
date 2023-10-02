import logging
from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class DataStrategy(ABC):
    """
    Abstract class for handling the data
    """

    @abstractmethod
    def handle_data(self, df: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass


class DataPreprocessingStrategy(DataStrategy):
    """
    Strategy for processing data
    """

    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        """
        Preprocessing the data

        Args:
            df<pd.Dataframe>: Pandas dataframe
        Returns:
            Union[pd.Dataframe,pd.Series]
        """
        try:
            to_drop = [
                "order_approved_at",
                "order_delivered_carrier_date",
                "order_delivered_customer_date",
                "order_estimated_delivery_date",
                "order_purchase_timestamp",
            ]
            data = data.drop(to_drop, axis=1)
            data = data.dropna()
            data["product_weight_g"].fillna(
                data["product_weight_g"].median(), inplace=True
            )
            data["product_length_cm"].fillna(
                data["product_length_cm"].median(), inplace=True
            )
            data["product_height_cm"].fillna(
                data["product_height_cm"].median(), inplace=True
            )
            data["product_width_cm"].fillna(
                data["product_width_cm"].median(), inplace=True
            )
            data["review_comment_message"].fillna("No review", inplace=True)

            data = data.select_dtypes(include=[np.number])
            cols_to_drop = ["customer_zip_code_prefix", "order_item_id"]
            data = data.drop(cols_to_drop, axis=1)

            return data
        except Exception as e:
            logging.info("Error occured in data preprocessing")
            raise (e)
