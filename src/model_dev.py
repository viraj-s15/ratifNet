import logging
from abc import ABC, abstractmethod

from sklearn.linear_model import LinearRegression
from pytorch_tabnet.tab_model import TabNetRegressor
from callbacks.mlflow_callbacks import MLCallback 

class Model(ABC):
    """
    Abstract class for all the models
    """

    @abstractmethod
    def train(self, X_train, y_train):
        pass


class LinearRegressionModel(Model):
    def train(self, X_train, y_train, **kwargs):
        try:
            reg = LinearRegression(**kwargs)
            reg.fit(X_train, y_train)
            logging.info("Model training complete")
            return reg
        except Exception as e:
            logging.info("An error occured in the Linear Regression model")
            raise e
        
        
class TabnetModel(Model):
    def train(self, X_train, y_train, **kwargs):
        try:
            mlcbck = MLCallback()
            reg = TabNetRegressor(verbose=0,seed=42)
            reg.fit(X_train=X_train, y_train=y_train,
              patience=300, max_epochs=2000,
              eval_metric=['rmse'], callbacks=[mlcbck])
            return reg
        except Exception as e:
            logging.info("An error occured in the Linear Regression model")
            raise e
