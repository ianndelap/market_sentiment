from google.cloud import storage
from google.cloud.storage import bucket
from google.resumable_media.requests import upload
from termcolor import colored
import pandas as pd
import joblib
import os

BUCKET_NAME='market-data-701-delap'

BUCKET_TRAIN_DATA_PATH = 'data/dump.json'

MODEL_NAME = "market_sentiment"
# google is creating this for us and model must be equal name across border
STORAGE_LOCATION = 'model/market_sentiment/versions/modelintel.joblib'


def get_data_from_gcp(nrows=1000, optimize=False, lines=True):
    """method to get the training data (or a portion of it) from google cloud bucket"""
    # Add Client() here
    client = storage.Client()
    path = f"gs://{BUCKET_NAME}/{BUCKET_TRAIN_DATA_PATH}"
    dftest = pd.read_json(path, nrows=nrows,lines=lines )
    return dftest

def upload_model_to_gcp(model_name):
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(STORAGE_LOCATION)
    blob.upload_from_filename(model_name)
    print('Success!')

def save_model(trained_model):
    """method that saves the model into a .joblib file and uploads it on Google Storage /models folder
    HINTS : use joblib library and google-cloud-storage"""
    # saving the trained model to disk is mandatory to then beeing able to upload it to storage
    joblib.dump(trained_model, 'modelintel.joblib')
    print(colored("modelintel.joblib saved locally", "green"))

# download the model to be predicted on
def download_model(bucket=BUCKET_NAME):
    client = storage.Client().bucket(bucket)
    blob = client.blob(STORAGE_LOCATION)
    blob.download_to_filename('modelintel.joblib')
    print(colored("=> model downloaded from storage", "green"))
    model = joblib.load('modelintel.joblib')
    # return our model to be used to predict!
    return model
# this guy underneath is causing this
# prediction = download_model(bucket= BUCKET_NAME)

if __name__ == '__main__':
    pass
    # model = save_model('modelintel.joblib')
    # upload_model_to_gcp(model)
