import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime

st.title('Market Analysis Based Off Twitter Sentiment')

# load the data frame
df = pd.DataFrame({
  'ticker option': ['$GE', '$BYND', '$INTC', '$BTC']
})
# load the line chart
chart_data = pd.DataFrame(
     np.random.randn(20, 4),
     columns=['$GE', '$INTC', '$BYND', '$BTC'])
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
analysis.metric("Analysis", "Bullish", "+23%")
