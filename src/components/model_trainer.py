import os
import sys
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from src.utils import (save_object,evaluate_model)

from catboost import CatBoostRegressor
from sklearn.ensemble import (
RandomForestRegressor, 
GradientBoostingRegressor,
AdaBoostRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'trained_model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("splitting the data into train and test")
            X_train,y_train,X_test,y_test =(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models ={
                "Random Forest": RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "AdaBoost": AdaBoostRegressor(),
                "Linear Regression": LinearRegression(),
                "KNN": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "XGBoost": XGBRegressor(),
                "CatBoost": CatBoostRegressor(verbose=False)
                }
            
            params = {
                    "Random Forest": {
                        "n_estimators": [8,16,32,64,128,256],
                    },
                    "Gradient Boosting": {
                        "n_estimators": [8,16,32,64,128,256],
                        "learning_rate": [.1, .01, .05, .001],
                        "sub_sample": [0.6,0.7,0.75,0.8,0.85,0.9],
                    },
                    "AdaBoost": {
                        "n_estimators": [8, 16, 32, 64, 128, 256],
                        "learning_rate": [0.1,0.01 ,0.5, .001]
                    },
                    "Linear Regression": {},
                    "KNN": {
                        "n_neighbors": [5, 7, 9, 11],
                    },
                    "Decision Tree": {
                        "criterion": ["squared_error", "friedman_mse", "absolute_error", "poisson"]
                    },
                    "XGBoost": {
                        "n_estimators": [8,16,32,64,128,256],
                        "learning_rate": [.1, .01, .05, .001],
                           },
                    "CatBoost": {
                        "iterations": [30, 50, 100],
                        "depth": [6, 8, 10],
                        "learning_rate": [0.01, 0.05, 0.1],
                    }
                }

            

            model_report: dict = evaluate_model(X_train=X_train,y_train=y_train,
                                                X_test=X_test,y_test=y_test,models= models,param = params)
            
            #getting the best model score
            best_model_score = max(sorted(model_report.values()))

            if best_model_score < 0.6:
                raise CustomException("All models performed poorly",sys)

            #best modal name
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
                ]
            
            best_model = models[best_model_name]

            logging.info(f"Best model is {best_model_name} with score {best_model_score}")

            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj= best_model
                
                )
            
            predicted = best_model.predict(X_test)

            r2_square = r2_score(y_test,predicted)

            return r2_square

        
            
        except Exception as e:
            raise CustomException(e,sys)

   
