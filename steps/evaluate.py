import logging

import pandas as pd
from zenml import step


@step
def evaluate(df: pd.DataFrame) -> None:
    pass
