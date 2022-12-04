from flask import request
from utils import dataload
import requests

X_test = dataload("df_feature_engineering.csv")


# Change models here
# request = {"workspace": "anshitasaxena","registry_name": "tuned-xgboost-model","model_name": "tuned_xgboost.json","version": "1.0.0"}
# request = {"workspace": "anshitasaxena","registry_name": "base-xgboost-model","model_name": "base_xgboost.json","version": "1.0.0"}
# request = {"workspace": "anshitasaxena","registry_name": "feature-selection-xgboost-model","model_name": "feature_selection_xgboost.json","version": "1.0.0"}
# r = requests.post("http://127.0.0.1:5000/download_registry_model",json=request)

# Get logs
# r = requests.get("http://127.0.0.1:5000/logs")

# Predict
r = requests.post("http://127.0.0.1:5000/predict", json=X_test.to_json())