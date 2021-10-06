"""
canonical imports here
"""

from market_sentiment.gcp import get_data_from_gcp
from numpy.lib.ufunclike import fix
from market_sentiment import auth
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
# import seaborn as sns
import tensorflow_hub as hub
import numpy as np
import tensorflow_text
import datetime as dt
import re
import yfinance as yf
import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="../credentials.json"


# GCP = URL replace with mine
# """Import Data
#     Note in the packages we eventually don't limit the rows to 1000 we'll need to change this to a google cloud URL due to data being massive"""
# def get_data(nrows=5000):
#     dftest = pd.read_json('/Users/munjikahalah/code/market_sentiment/raw_data/dump.json',lines=True, nrows=nrows)
#     return dftest

dftest = get_data_from_gcp(nrows=1000, optimize=False, lines=True)
# dftest = get_live_tweets()

# let's choose only the features we need
def grab_features(dftest):
    df = dftest[['full_text',
    'lang',
    'datetimestamp',
    'reply_count',
    'retweet_count',
    'quote_count',
    'created_at',
    'favorite_count',
    'symbols',
    'hashtags']]
    return df

# Sort out \n \r \r\n in full_text column
def replacen(x):
    return x.replace('\n','')
def replacer(x):
    return x.replace('\r','')
def replacern(x):
    return x.replace('\r\n','')

def clean_data(df):
    df['full_text'] = df['full_text'].apply(replacer)
    df['full_text'] = df['full_text'].apply(replacen)
    df['full_text'] = df['full_text'].apply(replacern)
    return df

# """
# Categorize the symbol and hashtag mentions for the four main culprits of this project:
# - Intel (intc)
# - Beyond Meat (bynd)
# - General Electric (ge)
# - Bitcoin (btc)
# """
# Intel
def tickerintc(x):
    if type(x) == list:
        newlist = []
        for item in x:
            newlist.append(item.lower())
        if 'intc' in newlist:
            return 1
        else:
            return 0

def hashtag_intc(x):
    if type(x) == list:
        newlist = []
        for item in x:
            newlist.append(item.lower())
        intc = ['intel','intc']
        if intc[0] in newlist or intc[1] in newlist:
            return 1
        else:
            return 0

# Beyond Meat
def tickerbynd(x):
    if type(x) == list:
        newlist = []
        for item in x:
            newlist.append(item.lower())
        if 'bynd' in newlist:
            return 1
        else:
            return 0

def hashtag_bynd(x):
    if type(x) == list:
        newlist = []
        for item in x:
            newlist.append(item.lower())
        bynd = ['beyondmeat','bynd']
        if bynd[0] in newlist or bynd[1] in newlist:
            return 1
        else:
            return 0
# General Electric
def tickerge(x):
    if type(x) == list:
        newlist = []
        for item in x:
            newlist.append(item.lower())
        if 'ge' in newlist:
            return 1
        else:
            return 0

def hashtag_ge(x):
    if type(x) == list:
        newlist = []
        for item in x:
            newlist.append(item.lower())
        ge = ['generalelectric','ge']
        if ge[0] in newlist or ge[1] in newlist:
            return 1
        else:
            return 0

# Bitcoin (  >:(   )
def tickerbtc(x):
    if type(x) == list:
        newlist = []
        for item in x:
            newlist.append(item.lower())
        if 'btc' in newlist:
            return 1
        else:
            return 0

def hashtag_btc(x):
    if type(x) == list:
        newlist = []
        for item in x:
            newlist.append(item.lower())
        btc = ['bitcoin','btc']
        if btc[0] in newlist or btc[1] in newlist:
            return 1
        else:
            return 0
##########################################
"""Create our new one-hot encoded ticker columns and fill nans with 0"""
def custom_ohe(df):
    df['intc_ticker'] = df['symbols'].apply(tickerintc)
    df['bynd_ticker'] = df['symbols'].apply(tickerbynd)
    df['ge_ticker'] = df['symbols'].apply(tickerge)
    df['btc_ticker'] = df['symbols'].apply(tickerbtc)

    # df['#intc_ticker'] = df['hashtags'].apply(hashtag_intc)
    # df['#bynd_ticker'] = df['hashtags'].apply(hashtag_bynd)
    # df['#ge_ticker'] = df['hashtags'].apply(hashtag_ge)
    # df['#btc_ticker'] = df['hashtags'].apply(hashtag_btc)
    return df

# """
# fill nan values with zeros in our new columns
# """
def remove_dollar(l):
    new_list = []
    for item in l:
        new_list.append(item.replace('$',''))
    return new_list

def impute(df):
    df['intc_ticker'].fillna(0.,inplace=True)
    df['bynd_ticker'].fillna(0.,inplace=True)
    df['ge_ticker'].fillna(0.,inplace=True)
    df['btc_ticker'].fillna(0.,inplace=True)

    # df['#intc_ticker'].fillna(0.,inplace=True)
    # df['#bynd_ticker'].fillna(0.,inplace=True)
    # df['#ge_ticker'].fillna(0.,inplace=True)
    # df['#btc_ticker'].fillna(0.,inplace=True)
    return df
# """
# remove nans from ticker colums
# """
# def remove_na(df):
#     df[['intc_ticker','bynd_ticker','ge_ticker','btc_ticker']].fillna(0.,inplace=True)
#     df[['#intc_ticker','#bynd_ticker','#ge_ticker','#btc_ticker']].fillna(0.,inplace=True)
#     return df

"""
fix date time set to day without specific times
"""
def fix_date_time(df):
    if df['created_at'].dtype == 'object':
        df['created_at'] = pd.to_datetime(df['created_at'])
    df['created_at'] = df['created_at'].dt.date
    return df

def drop_columns(df):
    df.drop(['favorite_count',
         'lang',
         'datetimestamp',
         'reply_count',
         'retweet_count',
         'quote_count',
         'quote_count',
         'symbols',
         'hashtags'],axis=1,inplace=True)
    return df

"""
remove links away from full_text
"""
def remove_urls(text):
    text = re.sub(r'https?://\S+', '', text)
    return text
def remove_url(df):
    df['full_text'] = df['full_text'].apply(remove_urls)
    return df

"""
create a vextor embedding matrix per tweet into a new dataframe variable vector_df
"""
def embed_matrix(df):
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual/3")
    text_result = embed(df['full_text'])
    vector_df = pd.DataFrame(text_result.numpy())
    return vector_df



"""
concat the two dataframes one against another to create one single dataframe
"""
def concat_vectors(df):
    df = pd.concat([df,embed_matrix(df)], axis=1, join="inner")
    return df

"""
select the dataframes for when the symbol tickers have a hit
"""
def symbols_hit(df, ticker=''):
    if ticker == 'INTC':
        intc_ticker_df = df[df['intc_ticker']==1.]
        return intc_ticker_df
    elif ticker =='BYND':
        bynd_ticker_df = df[df['bynd_ticker']==1.]
        return bynd_ticker_df
    elif ticker == 'GE':
        ge_ticker_df = df[df['ge_ticker']==1.]
        return ge_ticker_df
    else:
        ticker = 'BTC'
        btc_ticker_df = df[df['btc_ticker']==1.]
        return btc_ticker_df

"""
select the dataframes for when the hashtag tickers have a hit
"""
# def hashtag_hit(df, ticker=''):
#     if ticker == 'INTC':
#         intc_hash_ticker_df = df[df['#intc_ticker']==1.]
#         return intc_hash_ticker_df
#     elif ticker == 'BYND':
#         bynd_hash_ticker_df = df[df['#bynd_ticker']==1.]
#         return bynd_hash_ticker_df
#     elif ticker == 'GE':
#         ge_hash_ticker_df = df[df['#ge_ticker']==1.]
#         return ge_hash_ticker_df
#     else:
#         ticker = 'BTC'
#         btc_hash_ticker_df = df[df['#btc_ticker']==1.]
#         return btc_hash_ticker_df

def average_symbols_of_tweets_per_day(df, ticker=''):
    if ticker =='INTC':
        intc_ticker_mean_df = symbols_hit(df, 'INTC').groupby('created_at').mean().reset_index()
        print(f'INTC TICKER MEAN DF SHAPE =========== {intc_ticker_mean_df.shape}')
        return intc_ticker_mean_df
    elif ticker == 'BYND':
        bynd_ticker_mean_df = symbols_hit(df, 'BYND').groupby('created_at').mean().reset_index()
        print(f'BYND TICKER MEAN DF SHAPE =========== {bynd_ticker_mean_df.shape}')
        return bynd_ticker_mean_df
    elif ticker == 'GE':
        ge_ticker_mean_df = symbols_hit(df, 'GE').groupby('created_at').mean().reset_index()
        print(f'GE TICKER MEAN DF SHAPE =========== {ge_ticker_mean_df.shape}')
        return ge_ticker_mean_df
    else:
        ticker ='BTC'
        btc_ticker_mean_df = symbols_hit(df,'BTC').groupby('created_at').mean().reset_index()
        print(f'BTC TICKER MEAN DF SHAPE =========== {btc_ticker_mean_df.shape}')
        return btc_ticker_mean_df

# def average_hashtag_of_tweets_per_day(df, ticker=''):
#     if ticker =='INTC':
#         intc_hash_ticker_mean_df = hashtag_hit(df,'INTC').groupby('created_at').mean()
#         return intc_hash_ticker_mean_df
#     elif ticker == 'BYND':
#         bynd_hash_ticker_mean_df = hashtag_hit(df, 'BYND').groupby('created_at').mean()
#         return bynd_hash_ticker_mean_df
#     elif ticker == 'GE':
#         ge_hash_ticker_mean_df = hashtag_hit(df, 'GE').groupby('created_at').mean()
#         return ge_hash_ticker_mean_df
#     else:
#         ticker ='BTC'
#         btc_hash_ticker_mean_df = hashtag_hit(df,'BTC').groupby('created_at').mean()
#         return btc_hash_ticker_mean_df
"""
convert created_at index from object to datetime
"""
# symbols tickers dfs
def df_to_datetime(df):
    df.index = pd.to_datetime(df.index)
    return df

""" Import target Dataframe
    Convert tickers to 1 and 0 as target output series
    """
def download_yahoo_stocks(tickers= 'INTC BYND GE BTC', period = "6mo"):
    tickers = yf.download(tickers = tickers, interval='1d',period = period)

    INTCclose = tickers['Close']['INTC']
    BYNDclose = tickers['Close']['BYND']
    GEclose = tickers['Close']['GE']
    BTCclose = tickers['Close']['BTC']

    # return {"BTC": BTCclose, "BYND": BYNDclose,

    return INTCclose, BYNDclose, GEclose, BTCclose

def onezero(x):
    if x>0:
        return 1
    else:
        return 0

# Convert tickers to 1 and 0 as target output series
# we're taking NAN values and forcing it to be zero - (may need to discuss this)
def convert_tickers(INTCclose,BYNDclose, GEclose, BTCclose):
    intc_target_df = pd.Series(INTCclose).diff().apply(onezero)
    bynd_target_df = pd.Series(BYNDclose).diff().apply(onezero)
    ge_target_df = pd.Series(GEclose).diff().apply(onezero)
    btc_target_df = pd.Series(BTCclose).diff().apply(onezero)

    return intc_target_df, bynd_target_df, ge_target_df, btc_target_df

"""
Merge ticker feature selected aggredated dataframe with target target dataframe created
"""
def merge_ticker_with_target_dataframe(df_INTC, df_BYND, df_GE, df_BTC, intc_target_df, bynd_target_df, ge_target_df, btc_target_df):
    # print(f'MERGE TICKER SHAPES ============ {df_INTC.shape}')
    # print(f'MERGE TICKER SHAPES ============ {intc_target_df.shape}')
    INTC_df = pd.merge(df_INTC, intc_target_df,left_index=True,right_index=True)
    print(f'INTC DF SHAPE =================== {INTC_df.shape}')


    BYND_df = pd.merge(df_BYND, bynd_target_df,left_index=True,right_index=True)

    print(f'BYND DF SHAPE =================== {BYND_df.shape}')


    GE_df = pd.merge(df_GE, ge_target_df,left_index=True,right_index=True)
    print(f'GE DF SHAPE =================== {GE_df.shape}')

    BTC_df = pd.merge(df_BTC, btc_target_df,left_index=True,right_index=True)
    print(f'BTC DF SHAPE =================== {BTC_df.shape}')

    return INTC_df, BYND_df, GE_df, BTC_df

def find_tickers(text):
    return re.findall(r'[$][A-Za-z][\S]*', text)

def get_live_tweets():
    tweets_GE_list = []
    tweets_BYND_list = []
    tweets_INTC_list = []
    tweets_BTC_list = []

    tweets_GE = auth.api.search(q ="$GE", count = 1000)
    tweets_BYND = auth.api.search(q ="$BYND", count = 1000)
    tweets_INTC = auth.api.search(q ="$INTC", count = 1000)
    tweets_BTC = auth.api.search(q ="$BTC", count = 1000)

    json_data_GE = [r._json for r in tweets_GE]
    json_data_BYND = [r._json for r in tweets_BYND]
    json_data_BTC = [r._json for r in tweets_BTC]
    json_data_INTC = [r._json for r in tweets_INTC]
    # use cmnd key to highlight eveyrthing
    for tweet in json_data_GE:
        tweets_GE_list.append({'created_at': tweet['created_at'], 'full_text': tweet['text']})
    for tweet in json_data_BYND:
        tweets_BYND_list.append({'created_at': tweet['created_at'], 'full_text': tweet['text']})
    for tweet in json_data_BTC:
        tweets_BTC_list.append({'created_at': tweet['created_at'], 'full_text': tweet['text']})
    for tweet in json_data_INTC:
        tweets_INTC_list.append({'created_at': tweet['created_at'], 'full_text': tweet['text']})

    concat_tweets = tweets_GE_list +  tweets_BYND_list + tweets_INTC_list + tweets_BTC_list
    final_df = pd.DataFrame.from_dict(concat_tweets)
    final_df['symbols'] = final_df['full_text'].apply(find_tickers)
    final_df['symbols'] = final_df['symbols'].apply(remove_dollar)
    return final_df

def grab_live_tweets_to_vectorize(final_df):
    df = final_df
    df = clean_data(df)
    df = custom_ohe(df)
    df = impute(df)
    df = fix_date_time(df)
    df = remove_url(df)
    df = concat_vectors(df)
    df = df_to_datetime(df)
    df_INTC = average_symbols_of_tweets_per_day(df, ticker='INTC')
    df_INTC = df_INTC.set_index(['created_at'])
    df_BYND= average_symbols_of_tweets_per_day(df, ticker='BYND')
    df_BYND = df_BYND.set_index(['created_at'])
    df_GE= average_symbols_of_tweets_per_day(df, ticker='GE')
    df_GE = df_GE.set_index(['created_at'])
    df_BTC = average_symbols_of_tweets_per_day(df, ticker='BTC')
    df_BTC = df_BTC.set_index(['created_at'])

    INTCclose, BYNDclose, GEclose, BTCclose = download_yahoo_stocks(tickers= 'INTC BYND GE BTC', period='6mo')
    intc_target_df, bynd_target_df, ge_target_df, btc_target_df = convert_tickers(INTCclose, BYNDclose, GEclose, BTCclose)



    # print(f'INTC TARGET SHAPE ========== {intc_target_df.shape}')
    # print(f'BYND TARGET SHAPE ========== {bynd_target_df.shape}')

# merge below
    INTC_df, BYND_df, GE_df, BTC_df = merge_ticker_with_target_dataframe(
                                        df_INTC, df_BYND, df_GE, df_BTC,
                                        intc_target_df, bynd_target_df, ge_target_df, btc_target_df)
    return INTC_df, BYND_df, GE_df, BTC_df

    # print(INTC_df)
    # print(BYND_df)
    # print(GE_df)
    # print(BTC_df)

def grab_stocks(df):
    df = dftest
    df = grab_features(df)
    # print('grab_features==========================', df.shape)
    df = clean_data(df)
    # print('clean_data==========================', df.shape)
    df = custom_ohe(df)
    # print('custom_ohe==========================', df.shape)
    df = impute(df)
    # print('impute==========================', df.shape)
    # df = remove_na(df)
    df = fix_date_time(df)
    # print('fix_date_time==========================', df.shape)
    df = drop_columns(df)
    # print('drop_columns==========================', df.shape)
    df = remove_url(df)
    # print('remove_url==========================', df.shape)
    df = concat_vectors(df)
    # print('concat_vectors==========================', df.shape)
    # df = symbols_hit(df, ticker='BYND')
    # print('symbols_hit==========================', df.shape)
    # df = hashtag_hit(df, ticker='BYND')
    # print('hashtag_hit==========================', df.shape)
    df = df_to_datetime(df)

    df_INTC = average_symbols_of_tweets_per_day(df, ticker='INTC')
    df_BYND= average_symbols_of_tweets_per_day(df, ticker='BYND')
    df_GE= average_symbols_of_tweets_per_day(df, ticker='GE')
    df_BTC = average_symbols_of_tweets_per_day(df, ticker='BTC')

    # print('average_symbols==========================', df.shape)
    # print(df.columns)
    # df = average_hashtag_of_tweets_per_day(df, ticker='BYND')
    # print('average_hashtag==========================', df.shape)
    df_INTC = df_INTC.set_index(['created_at'])

    df_BYND = df_BYND.set_index(['created_at'])

    df_GE = df_GE.set_index(['created_at'])

    df_BTC = df_BTC.set_index(['created_at'])


    INTCclose, BYNDclose, GEclose, BTCclose = download_yahoo_stocks(tickers= 'INTC BYND GE BTC', period='6mo')
    intc_target_df, bynd_target_df, ge_target_df, btc_target_df = convert_tickers(INTCclose, BYNDclose, GEclose, BTCclose)
    INTC_df, BYND_df, GE_df, BTC_df = merge_ticker_with_target_dataframe(
                                        df_INTC, df_BYND, df_GE, df_BTC,
                                        intc_target_df, bynd_target_df, ge_target_df, btc_target_df)

    return INTC_df, BYND_df, GE_df, BTC_df
    # print(INTC_df)
    # print(BYND_df)
    # print(GE_df)
    # print(BTC_df)
if __name__ == '__main__':
    # df = dftest
    our_live_df = get_live_tweets()
    INTC_df, BYND_df, GE_df, BTC_df = grab_live_tweets_to_vectorize(our_live_df)
    INTC_df.to_csv(index=True, path_or_buf='X_pred_INTC.csv')
    BYND_df.to_csv(index=True,path_or_buf='X_pred_BYND.csv')
    GE_df.to_csv(index=True,path_or_buf='X_pred_GE.csv')
    BTC_df.to_csv(index=True,path_or_buf='X_pred_BTC.csv')
