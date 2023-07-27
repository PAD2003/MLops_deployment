import random
import time
import pickle
import pandas as pd
import uvicorn
import yaml
from fastapi import FastAPI, Request
from pydantic import BaseModel

from problem_config import create_prob_config
from raw_data_processor import RawDataProcessor

from evidently.test_suite import TestSuite
from evidently import ColumnMapping
from evidently.test_preset import NoTargetPerformanceTestPreset

class Data(BaseModel):
    id: str
    rows: list
    columns: list

class ModelPredictor:
    def __init__(self, config_file_path):
        with open(config_file_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.prob_config = create_prob_config(
            self.config["phase_id"], self.config["prob_id"]
        )

        # load category_index
        self.category_index = RawDataProcessor.load_category_index(self.prob_config)

        # load model
        with open(self.config["weight_path"], 'rb') as file:
            self.model = pickle.load(file)

    def detect_drift(self, feature_df) -> int:
        # read reference dataset
        train_x, train_y = RawDataProcessor.load_train_data(self.prob_config)

        column_mapping = ColumnMapping()

        column_mapping.numerical_features = self.prob_config.numerical_cols
        column_mapping.categorical_features = self.prob_config.categorical_cols

        tests = TestSuite(tests=[
            NoTargetPerformanceTestPreset(),
        ])

        tests.run(current_data=train_x, reference_data=feature_df,
                  column_mapping=column_mapping)
        output_tests = tests.as_dict()
        test_must_not_failed = ['Share of Out-of-List Values',
                                'Column Types', 'Share of Out-of-Range Values']
        for test in output_tests['tests']:
            if test['name'] in test_must_not_failed and test['status'] == 'FAIL':
                return 1
        
        return 0

    def predict(self, data: Data):
        # preprocess
        raw_df = pd.DataFrame(data.rows, columns=data.columns)
        feature_df = RawDataProcessor.apply_category_features(
            raw_df=raw_df,
            categorical_cols=self.prob_config.categorical_cols,
            category_index=self.category_index,
        )
        # # save request data for improving models
        # ModelPredictor.save_request_data(
        #     feature_df, self.prob_config.captured_data_dir, data.id
        # )

        prediction = self.model.predict(feature_df)
        labels_rank = {0: 'Normal',
               1: 'Other',
               2: 'Information Gathering',
               3: 'Denial of Service',
               4: 'Exploits',
               5: 'Malware'
              }
        prediction = [labels_rank[i] for i in prediction]
        is_drifted = self.detect_drift(feature_df)

        if not isinstance(prediction, list):
            prediction = prediction.tolist()
        return {
            "id": data.id,
            "predictions": prediction,
            "drift": is_drifted,
        }
    
    # @staticmethod
    # def save_request_data(feature_df: pd.DataFrame, captured_data_dir, data_id: str):
    #     if data_id.strip():
    #         filename = data_id
    #     else:
    #         filename = hash_pandas_object(feature_df).sum()
    #     output_file_path = os.path.join(captured_data_dir, f"{filename}.parquet")
    #     feature_df.to_parquet(output_file_path, index=False)
    #     return output_file_path

class PredictorApi:
    def __init__(self, predictor: ModelPredictor):
        self.predictor = predictor
        self.app = FastAPI()

        @self.app.get("/")
        async def root():
            return {"message": "hello"}

        @self.app.post("/phase-2/prob-2/predict")
        async def predict(data: Data, request: Request):
            # self._log_request(request)
            response = self.predictor.predict(data)
            # self._log_response(response)
            return response

    # @staticmethod
    # def _log_request(request: Request):
    #     pass

    # @staticmethod
    # def _log_response(response: dict):
    #     pass

    def run(self, port):
        uvicorn.run(self.app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    predictor = ModelPredictor(config_file_path="data/model_config/phase-2/prob-2/model-1.yaml")
    api = PredictorApi(predictor)
    api.run(port=5040)