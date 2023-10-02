import logging
from dataclasses import dataclass

import pandas as pd
from zenml import step


@dataclass
class IngestData:
    """
    Class for ingesting the csv data
    Would ideally be something like a sql db
    """

    data_path: str

    def get_data(self):
        logging.info(f"Ingesting data from {self.get_data}")
        return pd.read_csv(self.data_path)


@step
def ingest_data(data_path: str) -> pd.DataFrame:
    """
    Ingesting data from data_path

    Args:
        data_path<str>: path to the csv file
    Return:
        pandas dataframe
    """
    try:
        ingest_data = IngestData(data_path)
        df = ingest_data.get_data()
        return df
    except Exception as e:
        logging.error("Error while ingesting the data")
        raise e
