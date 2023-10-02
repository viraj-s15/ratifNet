from zenml import pipeline, pipelines

from steps.cleaning_data import clean_data
from steps.evaluate import evaluate
from steps.ingest_data import ingest_data
from steps.train import train


@pipeline
def training_pipeline(data_path: str):
    df = ingest_data(data_path)
    clean_data(df)
    train(df)
    evaluate(df)
