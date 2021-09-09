from math import sqrt
import pandas as pd
import joblib

def predict_model(ticker):
    model = joblib.load(f'model_{ticker}.joblib')
    X_pred = f'X_pred_{ticker}.csv'
    X_pred = pd.read_csv(X_pred, index_col=0)
    # print(X_pred)
    prediction = model.predict(X_pred)
    return prediction

if __name__ == '__main__':
    pass
    # predict_BTC = predict_model('BTC')
    # predict_GE = predict_model('GE')
    # predict_INTC = predict_model('INTC')
    # predict = predict_model("INTC")
    # print(predict_BTC)
    # print(type(predict_BTC))
