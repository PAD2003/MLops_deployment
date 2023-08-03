import random
import time
import pickle
import os
import pandas as pd
import uvicorn
import yaml
from fastapi import FastAPI, Request
from pydantic import BaseModel
from pandas.util import hash_pandas_object

from problem_config import create_prob_config
from raw_data_processor import RawDataProcessor

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
        # watch drift between coming requests and training data
        time.sleep(0.02)
        return random.choice([0, 1])

    def predict(self, data: Data):
        # preprocess
        raw_df = pd.DataFrame(data.rows, columns=data.columns)
        feature_df = RawDataProcessor.apply_category_features(
            raw_df=raw_df,
            categorical_cols=self.prob_config.categorical_cols,
            category_index=self.category_index,
        )
        # save request data for improving models
        ModelPredictor.save_request_data(
            feature_df, self.prob_config.captured_data_dir, data.id
        )
        
        prediction = self.model.predict(feature_df)
        is_drifted = self.detect_drift(feature_df)

        return {
            "id": data.id,
            "predictions": prediction.tolist(),
            "drift": is_drifted,
        }

    @staticmethod
    def save_request_data(feature_df: pd.DataFrame, captured_data_dir, data_id: str):
        if data_id.strip():
            filename = data_id
        else:
            filename = hash_pandas_object(feature_df).sum()
        output_file_path = os.path.join(captured_data_dir, f"{filename}.parquet")
        feature_df.to_parquet(output_file_path, index=False)
        return output_file_path

class PredictorApi:
    def __init__(self, predictor: ModelPredictor):
        self.predictor = predictor
        self.app = FastAPI()

        @self.app.get("/")
        async def root():
            return {"message": "hello"}

        @self.app.post("/phase-2/prob-1/predict")
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
    predictor = ModelPredictor(config_file_path="data/model_config/phase-2/prob-1/model-1.yaml")
    api = PredictorApi(predictor)
    api.run(port=5040)