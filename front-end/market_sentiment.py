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


st.markdown("""
<style>
.big-font {
    font-size:50px !important;
}
.medium-font {
    font-size:30px
}
</style>

""", unsafe_allow_html=True)

st.markdown('<p class="big-font">Market Analysis Based Off Twitter Sentiment</p>', unsafe_allow_html=True)


# st.title('Market Analysis Based Off Twitter Sentiment')

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
st.metric("Sentiment", 'Bullish', "+23%")
# st.write(response)

# load the line chart
plotly_figure = px.line(df, x=df['Date'], y=df['Price'], title=f'You Selected ${option}')
st.plotly_chart(plotly_figure)
# the metrics
st.metric("Stock Price",  df['Price'].iloc[-1], df['Price'].iloc[-1] - df['Price'].iloc[-2])
# if df['Price'].iloc[-2] - df['Price'].iloc[-1]< 0:
#     st.write('RED')
# else:
#     st.write('GREEN')
# st.write( df['Price'].iloc[-2] - df['Price'].iloc[-1])

# st.write(df['Price'].iloc[-2])
# st.write(df['Price'].iloc[-1])
