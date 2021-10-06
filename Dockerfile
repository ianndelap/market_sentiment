FROM python:3.8.6-buster

COPY api /api
COPY market_sentiment /market_sentiment
COPY requirements.txt /requirements.txt
COPY credentials.json /credentials.json
COPY model_BTC.joblib /model_BTC.joblib
COPY model_GE.joblib /model_GE.joblib
COPY model_INTC.joblib /model_INTC.joblib
COPY X_pred_INTC.csv /X_pred_INTC.csv
COPY X_pred_BYND.csv /X_pred_BYND.csv
COPY X_pred_GE.csv /X_pred_GE.csv
COPY X_pred_BTC.csv /X_pred_BTC.csv
RUN pip install --upgrade pip
RUN pip install -r requirements.txt


CMD uvicorn api.fast:app --host 0.0.0.0 --port $PORT
