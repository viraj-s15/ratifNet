import logging
from abc import ABC, abstractmethod

import optuna
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.linear_model import LinearRegression

from callbacks.mlflow_callbacks import MLCallback


class Model(ABC):
    """
    Abstract class for all the models
    """

    @abstractmethod
    def train(self, X_train, y_train):
        pass

    # @abstractmethod
    # def optimise(self,trial, x_train, y_train, x_test, y_test):
    #     pass
    #


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

    # def optimize(self, trial, x_train, y_train, x_test, y_test):
    #     try:
    #         n_estimators = trial.suggest_int("n_estimators", 1, 200)
    #         max_depth = trial.suggest_int("max_depth", 1, 20)
    #         min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
    #         reg = self.train(x_train, y_train, n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split)
    #         return reg.score(x_test, y_test)
    #     except Exception as e:
    #         logging.info(f"Error occured while optimizing Linear Regression model: {e}")
    #         raise e


class TabnetModel(Model):
    def train(self, X_train, y_train, **kwargs):
        try:
            mlcbck = MLCallback()
            reg = TabNetRegressor(verbose=0, seed=42)
            reg.fit(
                X_train=X_train,
                y_train=y_train,
                patience=200,
                max_epochs=2000,
                eval_metric=["rmse"],
                # callbacks=[mlcbck],
            )
            return reg
        except Exception as e:
            logging.info("An error occured in the Linear Regression model")
            raise e
        #


# def optimize(self, trial, x_train, y_train, x_test, y_test):
#     patience = trial.suggest_int("patience", 1, 300)
#     max_epochs = trial.suggest_int("max_epochs", 400, 2000)
#     reg = self.train(x_train, y_train, patience, max_epochs)
#     return reg.score(x_test, y_test)
