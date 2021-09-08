# # let’s create a root endpoint that will welcome the developers using our API.
# from market_sentiment.app import call_everything
# from pandas.core.frame import DataFrame
from market_sentiment.app import download_yahoo_stocks
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from market_sentiment.app import get_live_tweets, grab_live_tweets_to_vectorize
import numpy as np
import pandas as pd

app = FastAPI()

ticker_database = download_yahoo_stocks(tickers='INTC BYND GE BTC', period='1y')
sentiment_analysis_db = get_live_tweets()
sentiment_analysis_db = grab_live_tweets_to_vectorize(sentiment_analysis_db)

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
    period = {
        '1day': '1d',
        '3months': '3mo',
        '6months': '6mo',
        '1year': '1y'
    }
    name = {
        "bitcoin": 'BTC',
        "intel": 'INTC',
        "beyond_meat": 'BYND',
        "general_electric": 'GE'
    }
    if ticker == "BTC":
        index = 3
        period = period
        name = name['bitcoin']
        # sentiment_analysis_db[name]
    elif ticker == 'INTC':
        index = 0
        period = period
        name = name['intel']
        # sentiment_analysis_db[name]
    elif ticker == 'GE':
        index = 2
        period = period
        name = name['general_electric']
        # sentiment_analysis_db[name]
    else:
        index = 1
        period = period
        name = name['beyond_meat']
        # sentiment_analysis_db[name]

    df1 = (ticker_database[index]).where(pd.notnull(ticker_database[index]), 0)
    # sentiment_databse = (sentiment_analysis_db[name])

    return {'tickers': df1, "period": period}
