# import argparse
# import logging
import os
# import random
# import time
import pickle
# import mlflow
import pandas as pd
import uvicorn
import yaml
from fastapi import FastAPI, Request
from pandas.util import hash_pandas_object
from pydantic import BaseModel
# from frouros.detectors.data_drift import KSTest
# import numpy as np

from problem_config import ProblemConst, create_prob_config
from raw_data_processor import RawDataProcessor
# from utils import AppConfig, AppPath
import mlflow

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

        # load category_index
        self.category_index = RawDataProcessor.load_category_index(self.prob_config)

        # load model
        # with open(self.config["weight_path"], 'rb') as file:
        #     self.model = pickle.load(file)

        self.model = mlflow.pyfunc.load_model(self.config["weight_path"])

    def detect_drift(self, feature_df) -> int:
        # # watch drift between coming requests and training data
        # train_x, train_y = RawDataProcessor.load_train_data(self.prob_config)

        # train_x = train_x.to_numpy()
        # test_x = feature_df.to_numpy()
        
        # feature_idx = 0

        # while (feature_idx <= 40):
        #     if feature_idx == 2 or feature_idx == 3 or feature_idx == 4:
        #         feature_idx += 1
        #     else:
        #         alpha = 0.001
        #         detector = KSTest()
        #         _ = detector.fit(X=train_x[:, feature_idx])
        #         result, _ = detector.compare(X=test_x[:, feature_idx])

        #         if result.p_value <= alpha:
        #             print(f'Data drift detected at feature {feature_idx}')
        #             return 1
        #         else:
        #             feature_idx += 1

        return 0

    def predict(self, data: Data):
        # start_time = time.time()

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

        # run_time = round((time.time() - start_time) * 1000, 0)
        # logging.info(f"prediction takes {run_time} ms")
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

        @self.app.post("/phase-3/prob-1/predict")
        async def predict(data: Data, request: Request):
            self._log_request(request)
            response = self.predictor.predict(data)
            self._log_response(response)
            return response

    @staticmethod
    def _log_request(request: Request):
        pass

    @staticmethod
    def _log_response(response: dict):
        pass

    def run(self, port):
        uvicorn.run(self.app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    predictor = ModelPredictor(config_file_path="data/model_config/phase-3/prob-1/model-1.yaml")
    api = PredictorApi(predictor)
    api.run(port=5040)
