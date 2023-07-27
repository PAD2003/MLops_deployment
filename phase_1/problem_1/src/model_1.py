import pickle
import xgboost as xgb

model_path = 'model_1.pkl'

with open(model_path, 'rb') as file:
    model = pickle.load(file)

print(model)