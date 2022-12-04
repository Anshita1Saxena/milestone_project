"""
If you are in the same directory as this file (app.py), you can run run the app using gunicorn:
    
    $ gunicorn --bind 0.0.0.0:<PORT> app:app

gunicorn can be installed via:

    $ pip install gunicorn

"""
import os
from pathlib import Path
import logging
from flask import Flask, jsonify, request, abort
import sklearn
import pandas as pd
import joblib
from comet_ml import API
from dotenv import load_dotenv
import pickle
import xgboost as xgb

# from utils import dataload

load_dotenv()

LOG_FILE = os.environ.get("FLASK_LOG", "flask.log")
MODELS_DIR = "models_comet"
model = xgb.XGBClassifier()

app = Flask(__name__)


@app.before_first_request
def before_first_request():
    """
    Hook to handle any initialization before the first request (e.g. load model,
    setup logging handler, etc.)
    """
    # TODO: setup basic logging configuration
    logging.basicConfig(filename=LOG_FILE, level=logging.INFO)

    # TODO: any other initialization before the first request (e.g. load default model)
    global model
    workspace_name = "anshitasaxena"
    default_registry = "tuned-xgboost-model"
    default_model = "tuned_xgboost.json"
    default_model_dir = os.path.join(MODELS_DIR, default_model)
    request = {
        "workspace": workspace_name,
        "registry_name": default_registry,
        "version": "1.0.0",
        }
    if(not os.path.isfile(default_model_dir)):
        app.logger.info(f"Downloading the default model {default_model} fom CometML")
        API(api_key=os.getenv("COMET_API_KEY")).download_registry_model(**request, output_path=MODELS_DIR)
        
        if(not os.path.isfile(default_model_dir)):
            app.logger.info("Cannot download the model. Check the comet project and API key.")
        else:
            app.logger.info("Downloaded the model succesfully.")
            # model = pickle.load(open(default_model_dir, "rb"))
            model.load_model(default_model_dir)
    else:
        model.load_model(default_model_dir)
        app.logger.info(f"The default model {default_model} already exist. Load default model.")
    

@app.route("/logs", methods=["GET"])
def logs():
    """Reads data from the log file and returns them as the response
    
    Example of a request:
        r = requests.get("http://127.0.0.1:5000/logs")
    """
    
    # TODO: read the log file specified and return the data
    FILE_PATH = "flask.log"
    with open(FILE_PATH) as f:
        response = f.read().splitlines()

    return jsonify(response)  # response must be json serializable!


@app.route("/download_registry_model", methods=["POST"])
def download_registry_model():
    """
    Handles POST requests made to http://IP_ADDRESS:PORT/download_registry_model

    The comet API key should be retrieved from the ${COMET_API_KEY} environment variable.

    Recommend (but not required) json with the schema:

        {
            workspace: (required),
            model: (required),
            version: (required),
            ... (other fields if needed) ...
        }
    
    Example of request:
        requests = {"workspace": "anshitasaxena","registry_name": "randomforest","model_name": "randomforest.pkl","version": "1.0.0"}
        r = requests.post("http://0.0.0.0:5000/download_registry_model",json=request)
    
    """
    # Get POST json data
    json = request.get_json()
    app.logger.info(json)
    model_name = json["model_name"] # to get the pkl file such as "randomforest.pkl"
    model_dir = os.path.join(MODELS_DIR, model_name)
    global model

    # TODO: check to see if the model you are querying for is already downloaded
    if(os.path.isfile(model_dir)):
    # TODO: if yes, load that model and write to the log about the model change.  
    # eg: app.logger.info(<LOG STRING>)
        # model = pickle.load(open(os.path.join(MODELS_DIR, model_name), "rb"))
        model = xgb.XGBClassifier()
        model.load_model(model_dir)
        response = f"Model {model_name} has been downloaded already."
        app.logger.info(response)
    
    # TODO: if no, try downloading the model: if it succeeds, load that model and write to the log
    # about the model change. If it fails, write to the log about the failure and keep the 
    # currently loaded model
    else:
        app.logger.info(f"Model {model_name} is not found. Downloading the model from comet...")
        req = {
            "workspace": json["workspace"],
            "registry_name": json["registry_name"],
            "version": json["version"],
        }
        API(api_key=os.getenv("COMET_API_KEY")).download_registry_model(**req, output_path=MODELS_DIR)
        
        # check whether the model is downloaded successfully
        if(os.path.isfile(model_dir)):
            # model = pickle.load(open(os.path.join(MODELS_DIR, model_name), "rb"))
            model = xgb.XGBClassifier()
            model.load_model(model_dir)
            response = f"Model {model_name} has been downloaded successfully."
        else:
            response = f"Download failed. Currently loaded model is kept: {str(model)}"
            
    # Tip: you can implement a "CometMLClient" similar to your App client to abstract all of this
    # logic and querying of the CometML servers away to keep it clean here

    app.logger.info(response)
    return jsonify(response)  # response must be json serializable!


@app.route("/predict", methods=["POST"])
def predict():
    """
    Handles POST requests made to http://IP_ADDRESS:PORT/predict

    Returns predictions
    
    Example of a request:
        r = requests.post("http://127.0.0.1:5000/predict", json=X_test.to_json())
    """
    # Get POST json data
    json = request.get_json()

    # TODO:
    global model
    df_X_test = pd.read_json(json)
    response = model.predict_proba(df_X_test)[:, 1]
    
    response = pd.DataFrame(response).to_json()
    app.logger.info(response)
    return jsonify(response)  # response must be json serializable!

app.run(host='0.0.0.0', port=5000)
