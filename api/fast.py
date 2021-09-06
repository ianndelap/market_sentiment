# # let’s create a root endpoint that will welcome the developers using our API.
# from market_sentiment.app import call_everything
# from pandas.core.frame import DataFrame
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
# import pandas as pd
# from datetime import datetime
# import pytz
# import joblib
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)
# testing to see if we can fire up
# @app.get("/")
# def index():
#     return {"greeting": "Hello world"}

# @app.get("/deploy")
# def get_stocks():

#     return {'gr'}
