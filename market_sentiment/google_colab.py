import snscrape.modules.twitter as sntwitter
import pandas as pd
import tensorflow_hub as hub
import numpy as np
import tensorflow as tf
import tensorflow_text
import re
import yfinance as yf
from yfinance import ticker
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,VotingClassifier
from sklearn.model_selection import train_test_split,cross_validate
import joblib
from sklearn.model_selection import GridSearchCV

def replacen(x):
    return x.replace('\n','')
def replacer(x):
    return x.replace('\r','')
def replacern(x):
    return x.replace('\r\n','')

# Using TwitterSearchScraper to scrape data and append tweets to list
def get_tweets(tickers):
    tweets_list = []
    for i,tweet in enumerate(sntwitter.TwitterSearchScraper('${}'.format(tickers)).get_items()):
        if i > 1000:
            break
        tweets_list.append([tweet.date, tweet.content])
        tweet_df = pd.DataFrame(tweets_list, columns=['created_at', 'full_text'])
    return tweet_df
# function to take each full text and apply cleaning
def clean_data(df):
    df['full_text'] = df['full_text'].apply(replacer)
    df['full_text'] = df['full_text'].apply(replacen)
    df['full_text'] = df['full_text'].apply(replacern)
    return df
# remove dollar sign
def remove_dollar(l):
    new_list = []
    for item in l:
        new_list.append(item.replace('$',''))
    return new_list
# remove urls
def remove_urls(text):
    text = re.sub(r'https?://\S+', '', text)
    return text
# apply the remove url on the dataframe column
def remove_url(df):
    df['full_text'] = df['full_text'].apply(remove_urls)
    return df
# locate tickers
def find_tickers(text):
    return re.findall(r'[$][A-Za-z][\S]*', text)
# googles technology to give our text vector spaces
def embed_matrix(df):
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual/3")
    text_result = embed(df['full_text'])
    vector_df = pd.DataFrame(text_result.numpy())
    return vector_df

def df_to_datetime(df):
    df.index = pd.to_datetime(df.index)
    return df

def average_tweets_per_day(df):
    # print(df)
    ticker_mean_df = df.groupby('created_at').mean().reset_index()
    # print(f'TICKER MEAN DF SHAPE =========== {ticker_mean_df.shape}')
    return ticker_mean_df

def fix_date_time(df):
    if df['created_at'].dtype == 'object':
        df['created_at'] = pd.to_datetime(df['created_at'])
    df['created_at'] = df['created_at'].dt.date
    return df
# takes our embedded matrix and original dataframe and combines them
def concat_vectors(df):
    df = pd.concat([df,embed_matrix(df)], axis=1, join="inner")
    return df

# simply cleans our tweets dataframe, takes away /n and removes url
def clean_tweets(dataframe):
    cleaned_df = clean_data(dataframe)
    cleaned_df = remove_url(dataframe)
    return cleaned_df

# grabs the tweets dataframe and applies the concat vectors to merge our embedded matrix
# and original dataframe into one. Grabs average tweets per day and groups them by dates
# using the Mean of each tweet per day.
def grab_live_tweets_to_vectorize(dataframe):
    df = dataframe
    df = fix_date_time(df)
    df = concat_vectors(df) #concats our embedded matrix into the original dataframe
    df = df_to_datetime(df)
    df = average_tweets_per_day(df)
    df = df.set_index(['created_at'])
    return df

def download_yahoo_stocks(tickers= '', period = "6mo"):
    tickers = yf.download(tickers = tickers, interval='1d',period = period)
    tickers_close = tickers['Close']
    return tickers_close

def onezero(x):
    if x>0:
        return 1
    else:
        return 0
# creating our target on the stocks closed day
def convert_tickers(tickers_close):
    tickers_close = pd.Series(tickers_close).diff().apply(onezero)
    return tickers_close

def merge_ticker_with_target_dataframe(df, target_df):
#   print(f'Initial DF with vectors of tweets is =========== {df}')
#   print(f'above SHAPES is ============ {df.shape}')

#   print(f'Target DF of tickers with 1 or 0 on close is =========== {target_df}')
#   print(f'above TICKER SHAPES ============ {target_df.shape}')

  stock_df = pd.merge(df, target_df,left_index=True,right_index=True)

#   print(f'STOCK_DF DF SHAPE =================== {stock_df}')
#   print(f'STOCK_DF SHAPE =================== {stock_df.shape}')
  return stock_df

def ml_model(df):
    X = df.drop(columns='Close')
    y = df['Close']
    # split our X and y into training and tests sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    # instantiate the model --> open to changes on type of ML Algorithm
    voting_classifier_soft = VotingClassifier(estimators = [
        ('svc',SVC(probability=True)),
        ('rf',RandomForestClassifier()),
        ], voting='soft')
    cv_scores = cross_validate(voting_classifier_soft,X,y,cv=4,scoring=['f1','accuracy'])
    print(cv_scores)


def master_function(ticker, dataframe):
    pass


if __name__ == '__main__':
    ticker = 'BE'
    dataframe = get_tweets(ticker)
    dataframe = clean_tweets(dataframe)
    # print(dataframe)
    dataframe = grab_live_tweets_to_vectorize(dataframe)
    stock_price = download_yahoo_stocks(ticker, '6mo')
    stock_price = convert_tickers(stock_price)
    final_df = merge_ticker_with_target_dataframe(dataframe, stock_price)
    final_df_scores = ml_model(final_df)
    print(final_df_scores)
