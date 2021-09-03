import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime
import requests
import yfinance as yf

tickers = yf.download(tickers = 'INTC BYND GE BTC', interval='1d',start="2018-01-01", end="2021-08-01")
# market_url = "http://0.0.0.0:8000/"

# response = requests.get(
#     market_url
# ).json()

# # st.write(response)
# st.title(response["greeting"])


st.title('Market Analysis Based Off Twitter Sentiment')

# load the data frame
df = pd.DataFrame({
  'ticker option': ['$GE', '$BYND', '$INTC', '$BTC']
})
# load the line chart
chart_data = pd.DataFrame(
     np.random.randn(20, 1),
     columns=['$GE'])
st.line_chart(chart_data)


option = st.sidebar.selectbox(
    'Choose a $Ticker',
     df['ticker option'])
'You selected:', option

start_time = st.slider(
    "When do you start?",
     value=datetime(2021, 5, 2),
     format="MM/DD/YY")
st.write("Dates", start_time)

score, analysis = st.columns(2)
score.metric("Stock Price", "$125", "$2.45")
analysis.metric("Sentiment", "Bullish", "+23%")
