import os 
import sys
from dataclasses import dataclass

from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from src.logger import logging
from src.utils import save_object, evaluate_models
from src.exception import CustomException
from sklearn.metrics import r2_score


@dataclass
class ModelTrainerConfig:
    def trained_model_file_path(self):
        return os.path.join('artifacts', 'model.pkl')
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")

            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            # X_train, y_train, X_test, y_test = (
            #     train_array[:,:-1], #features for training
            #     train_array[:,-1], #target variable for training
            #     test_array[:,:-1], 
            #     test_array[:,-1] 
            # )
            
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbours Regressor": KNeighborsRegressor(),
                "XGBRegressor": XGBRegressor(),
                "Ada Boost Regressor": AdaBoostRegressor()
            }

            params = {
                "Decision Tree": {
                    "criterion": ["squared_error", "friedman_mse", "absolute_error", "poisson"]
                    # "splitter": ["best", "random"],
                    # "max_features": ["sqrt", "log2"]
                },

                "K-Neighbours Regressor": {
                    "n_neighbors": [3, 5, 7, 9, 11],
                    "weights": ["uniform", "distance"],
                    "algorithm": ["auto", "ball_tree", "kd_tree", "brute"]
                    # "leaf_size": [10, 20, 30, 40],
                    # "p": [1, 2]
                },

                "Random Forest": {
                    "n_estimators": [8, 16, 32, 64, 128, 256]
                    # "criterion": ["squared_error", "friedman_mse"],
                    # "max_features": ["sqrt", "log2", None]
                },

                "Gradient Boosting": {
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "subsample": [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                    # "loss": ["squared_error", "huber", "absolute_error", "quantile"],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],

                },

                "Linear Regression": {},

                "XGBRegressor": {
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "n_estimators": [8, 16, 32, 64, 128, 256]
                },

                # "Ada Boost Regressor":{
                #     'depth': [6,8,10],
                #     'learning_rate': [0.01, 0.05, 0.1],
                #     'iterations': [30, 50, 100]
                # },

                "Ada Boost Regressor": {
                    "learning_rate": [0.1, 0.01, 0.5, 0.001],
                    "n_estimators": [8, 16, 32, 64, 128, 256]
                },

                # "CatBoosting Regressor": {
                #     "learning_rate": [0.1, 0.01, 0.5, 0.001],
                #     "n_estimators": [8, 16, 32, 64, 128, 256]
                #     # "loss": ["linear", "square", "exponential"]
                # }
            }

            model_report = evaluate_models(X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test, models = models, param = params)
            logging.info(f"Model reports: {model_report}")

            #to get best score from dictionary
            best_model_score = max(sorted(model_report.values()))

            #to get the best model name 
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            #define threshold
            if best_model_score < 0.6:
                raise CustomException("No best model found")
            else:
                logging.info(f"Best model found: {best_model} with r2 score: {best_model_score}")

            #save object 
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path(),
                obj=best_model
            )

            #prediction
            predicted_result = best_model.predict(X_test)
            r2_square_value = r2_score(y_test, predicted_result)
            logging.info(f'The Probability r2 score is {r2_square_value}')
            logging.info('Final model saved successfully')
            return r2_square_value, best_model
        
        except Exception as e:
            raise CustomException(e, sys)   