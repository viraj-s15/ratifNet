from zenml import pipeline, pipelines

from steps.cleaning_data import clean_data
from steps.evaluate import evaluate
from steps.ingest_data import ingest_data
from steps.train import train_model


@pipeline
def training_pipeline(data_path: str):
    df = ingest_data(data_path)
    X_train, X_test, y_train, y_test = clean_data(df)
    model = train_model(X_train, X_test, y_train, y_test)
    r2_score, rmse = evaluate(model, X_test, y_test)
