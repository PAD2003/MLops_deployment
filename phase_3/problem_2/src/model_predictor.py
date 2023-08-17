# import argparse
import logging
import os
# import pickle
import time
import pandas as pd
import uvicorn
import yaml
import mlflow
from fastapi import FastAPI, Request
from pandas.util import hash_pandas_object
from pydantic import BaseModel

from problem_config import ProblemConst, create_prob_config
from raw_data_processor_prob_2 import RawDataProcessorProb2
# from utils import AppConfig, AppPath

# from evidently.test_suite import TestSuite
# from evidently import ColumnMapping
# from evidently.test_preset import NoTargetPerformanceTestPreset

PREDICTOR_API_PORT = 8000


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
        self.category_index = RawDataProcessorProb2.load_category_index(self.prob_config)
        self.model = mlflow.pyfunc.load_model(self.config["weight_path"])

    def predict(self, data: Data):
        start_time = time.time()

        # preprocess
        raw_df = pd.DataFrame(data.rows, columns=data.columns)
        feature_df = RawDataProcessorProb2.apply_category_features(
            raw_df=raw_df,
            categorical_cols=self.prob_config.categorical_cols,
            category_index=self.category_index,
        )
        # save request data for improving models
        ModelPredictor.save_request_data(
            feature_df, self.prob_config.captured_data_dir, data.id
        )

        prediction = self.model.predict(feature_df)
        labels_rank = {
            0: 'Normal',
            1: 'Other',
            2: 'Information Gathering',
            3: 'Denial of Service',
            4: 'Exploits',
            5: 'Malware'
        }
        prediction = [labels_rank[i] for i in prediction]

        run_time = round((time.time() - start_time) * 1000, 0)
        print(run_time)
        if not isinstance(prediction, list):
            prediction = prediction.tolist()
        return {
            "id": data.id,
            "predictions": prediction,
            "drift": 0,
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

        @self.app.post("/phase-3/prob-2/predict")
        async def predict(data: Data, request: Request):
            response = self.predictor.predict(data)
            return response

    def run(self, port):
        uvicorn.run(self.app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    predictor = ModelPredictor(config_file_path="data/model_config/phase-3/prob-2/model-1.yaml")
    api = PredictorApi(predictor)
    api.run(port=5040)
