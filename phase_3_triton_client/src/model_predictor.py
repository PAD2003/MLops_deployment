import pandas as pd
import uvicorn
import yaml
from fastapi import FastAPI, Request
from pydantic import BaseModel
from problem_config import create_prob_config
from raw_data_processor_prob_1 import RawDataProcessorProb1
from raw_data_processor_prob_2 import RawDataProcessorProb2
import tritonclient.http as httpclient

class Data(BaseModel):
    id: str
    rows: list
    columns: list

class ModelPredictorProb1:
    def __init__(self, config_file_path):
        with open(config_file_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.prob_config = create_prob_config(
            self.config["phase_id"], self.config["prob_id"]
        )
        self.category_index = RawDataProcessorProb1.load_category_index(self.prob_config)

    def predict(self, data: Data):
        # preprocess
        raw_df = pd.DataFrame(data.rows, columns=data.columns)
        feature_df = RawDataProcessorProb1.apply_category_features(
            raw_df=raw_df,
            categorical_cols=self.prob_config.categorical_cols,
            category_index=self.category_index,
        )
        # # save request data for improving models
        # ModelPredictorProb1.save_request_data(
        #     feature_df, self.prob_config.captured_data_dir, data.id
        # )

        # inference
        feature_df = feature_df.to_numpy(dtype='float32')
        triton_input = httpclient.InferInput('input', feature_df.shape, datatype="FP32")
        triton_input.set_data_from_numpy(feature_df)
        prediction = client.infer(model_name="xgboost_prob_1", inputs=[triton_input])
        prediction = prediction.as_numpy("label")[:,0]

        return {
            "id": data.id,
            "predictions": prediction.tolist(),
            "drift": 0,
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

class ModelPredictorProb2:
    def __init__(self, config_file_path):
        with open(config_file_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.prob_config = create_prob_config(
            self.config["phase_id"], self.config["prob_id"]
        )
        self.category_index = RawDataProcessorProb2.load_category_index(self.prob_config)

    def predict(self, data: Data):
        # preprocess
        raw_df = pd.DataFrame(data.rows, columns=data.columns)
        feature_df = RawDataProcessorProb2.apply_category_features(
            raw_df=raw_df,
            categorical_cols=self.prob_config.categorical_cols,
            category_index=self.category_index,
        )
        # # save request data for improving models
        # ModelPredictorProb2.save_request_data(
        #     feature_df, self.prob_config.captured_data_dir, data.id
        # )

        # inference
        feature_df = feature_df.to_numpy(dtype='float32')
        triton_input = httpclient.InferInput('input', feature_df.shape, datatype="FP32")
        triton_input.set_data_from_numpy(feature_df)
        prediction = client.infer(model_name="xgboost_prob_2", inputs=[triton_input])
        prediction = prediction.as_numpy("label")[:,0]

        labels_rank = {
            0: 'Normal',
            1: 'Other',
            2: 'Information Gathering',
            3: 'Denial of Service',
            4: 'Exploits',
            5: 'Malware'
        }
        prediction = [labels_rank[i] for i in prediction]
        if not isinstance(prediction, list):
            prediction = prediction.tolist()
        return {
            "id": data.id,
            "predictions": prediction,
            "drift": 0,
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
    def __init__(self, predictor_1: ModelPredictorProb1, predictor_2: ModelPredictorProb2):
        self.predictor_1 = predictor_1
        self.predictor_2 = predictor_2
        self.app = FastAPI()

        @self.app.post("/phase-3/prob-1/predict")
        async def predict(data: Data, request: Request):
            response = self.predictor_1.predict(data)
            return response

        @self.app.post("/phase-3/prob-2/predict")
        async def predict(data: Data, request: Request):
            response = self.predictor_2.predict(data)
            return response

    def run(self, port):
        uvicorn.run(self.app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    client = httpclient.InferenceServerClient(url="localhost:7999")
    predictor_1 = ModelPredictorProb1(config_file_path="data/model_config/model_prob_1.yaml")
    predictor_2 = ModelPredictorProb2(config_file_path="data/model_config/model_prob_2.yaml")
    api = PredictorApi(predictor_1, predictor_2)
    api.run(port=5040)
