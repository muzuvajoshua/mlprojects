import sys
import os
import pandas as pd
import numpy as np

from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            modal_path = "artifacts/trained_model.pkl"
            preprocessor_path = "artifacts/preprocessor.pkl"
            model = load_object(file_path= modal_path)
            preprocessor = load_object(file_path = preprocessor_path)
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e, sys)


# responsible for mapping my input data from the frontend to the backend
class CustomData:
    def __init__(self,
                 gender:str,
                 race_ethnicity:str,
                 parental_education,
                 lunch:str,
                 test_preparation:str,
                 reading_score:int,
                 writing_score:int):
        self.gender =gender
        self.race_ethnicity = race_ethnicity
        self.parental_education = parental_education
        self.lunch = lunch
        self.test_preparation = test_preparation
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_data_frame(self):
       try:
          custom_data_input_dict={
              "gender":self.gender,
              "race/ethnicity":self.race_ethnicity,
              "parental level of education":self.parental_education,
              "lunch":self.lunch,
                "test preparation course":self.test_preparation,
                "reading score":self.reading_score,
                "writing score":self.writing_score
          }

          return pd.DataFrame([custom_data_input_dict])
       except Exception as e:
          raise CustomException(e, sys)
