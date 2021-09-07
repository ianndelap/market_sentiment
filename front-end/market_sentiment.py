import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime
import requests
import yfinance as yf
import plotly.express as px
from datetime import datetime

# Some notes:
# 1.) market_url is our backend link that is being connected to uvcorn at the moment, using --reload it'll update changes
# that are made in the api/fast.py. then response is grabbing this to be displayed. We call our respone[key] to output
# the value.
# tickers = yf.download(tickers = 'INTC BYND GE BTC', interval='1d', )
market_url = "http://0.0.0.0:8001/deploy-stocks"


st.title('Market Analysis Based Off Twitter Sentiment')

# the side bar that allows us to choose the tickers and essentially acts as params
option = st.sidebar.selectbox(
    'Choose a $Ticker',
     ['GE', 'BYND', 'INTC', 'BTC'])

time_stamps = st.sidebar.selectbox(
    'Choose a time',
     ['1d', '3mo', '6mo', '1y'])

# our response to read our api
response = requests.get(
    market_url,
    params={"ticker": option, 'period': time_stamps}
).json()

# load our dataframe
df = pd.DataFrame({'Date': response['tickers'].keys(),'Price': response['tickers'].values()})
# the metrics
score, analysis = st.columns(2)

score.metric("Stock Price",  df['Price'].iloc[-1], df['Price'].iloc[-2] - df['Price'].iloc[-1])
analysis.metric("Sentiment", "Bullish", "+23%")


# load the line chart
plotly_figure = px.line(df, x=df['Date'], y=df['Price'], title=f'You Selected ${option}')
st.plotly_chart(plotly_figure)
