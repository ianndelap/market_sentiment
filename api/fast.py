# # let’s create a root endpoint that will welcome the developers using our API.
# from market_sentiment.app import call_everything
# from pandas.core.frame import DataFrame
from market_sentiment.predict import predict_model
from market_sentiment.app import download_yahoo_stocks
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from market_sentiment.app import get_live_tweets, grab_live_tweets_to_vectorize
import numpy as np
import pandas as pd

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)
# testing to see if we can fire up
@app.get("/")
def index():
    return {"greeting": "Phase 1"}

@app.get("/deploy-stocks")
def get_stocks(ticker='GE', period = '6mo'):
    if ticker == "BTC":
        index = 3
        period = period
    elif ticker == 'INTC':
        index = 0
        period = period
    elif ticker == 'GE':
        index = 2
        period = period
    else:
        index = 1
        period = period

    ticker_database = download_yahoo_stocks(tickers='INTC BYND GE BTC', period=period)

    df1 = (ticker_database[index]).where(pd.notnull(ticker_database[index]), 0)
    # grab prediction model
    messages = {
        '1': 'Twitter users are Bullish',
        '0': 'Twitter users are Bearish'
    }

    predict_INTC = int(predict_model('INTC')[0])
    predict_BTC = int(predict_model('BTC')[0])
    predict_GE = int(predict_model('GE')[0])
    if predict_GE == 1:
        prediction_GE = messages['1']
    else:
        prediction_GE = messages['0']

    if predict_BTC == 1:
        prediction_BTC = messages['1']
    else:
        prediction_BTC = messages['0']

    if predict_INTC == 1:
        prediction_INTC = messages['1']
    else:
        prediction_INTC = messages['0']

    predict = { 'GE':prediction_GE,
                'BTC' :prediction_BTC,
                "INTC": prediction_INTC
               }
    # if predict_BTC == 1:
    #     predict = 'Twitter users believe this is bullish'
    # elif predict_GE == 1:
    #     predict = 'Bullish, Twitter Users Agree this Stock is promising'
    # else:
    #     predict = 'Bearish, Twitter Users Agree to stay away from this stock'

    return {'tickers': df1, 'predict': predict }
