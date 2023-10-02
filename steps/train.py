import logging

import pandas as pd
from zenml import step


@step
def train(df: pd.DataFrame) -> None:
    pass
