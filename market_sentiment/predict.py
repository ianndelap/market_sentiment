from market_sentiment.gcp import MODEL_NAME
from market_sentiment.app import grab_stocks
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt
from market_sentiment.gcp import prediction


# def evaluate_model(y, y_pred):
#     MAE = round(mean_absolute_error(y, y_pred), 2)
#     RMSE = round(sqrt(mean_squared_error(y, y_pred)), 2)
#     res = {'MAE': MAE, 'RMSE': RMSE}
#     return res

def predit_model(model):
    y_predict = 'predicting our model'
    print( y_predict)


if __name__ == '__main__':
#    our_final_prediction =  predit_model(model= prediction)
#    print(our_final_prediction)
    pass
