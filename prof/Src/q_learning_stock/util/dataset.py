import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import datetime
import json
import os
import datetime as dt
import math
from pandas_datareader.av.time_series import AVTimeSeriesReader
import json
import config
import indicators

def get_train_test(df: pd.DataFrame):
    split_prop = config.train_split
    df_train, df_test = train_test_split(df, train_size=split_prop, test_size=1-split_prop, shuffle=False)
    print("Number of test samples = {len(df_train)}")
    x = df_train.loc[:,df.columns].values
    min_max_scaler = MinMaxScaler()
    x_train = min_max_scaler.fit_transform(x)
    x_test = min_max_scaler.transform(df_test.loc[:,df.columns])
    return x_train, x_test

def read_df(dataset_path: str) -> pd.DataFrame:
    """
    Reads csv data according to the column_names config and adds columns: day of week, week of year, month of year and year

    Note: Removes Date column
    """
    df = pd.read_csv(dataset_path, usecols=config.column_names, parse_dates=['Date'])
    #df['Date'].apply(convert_date_format)
    #df = df.drop(columns=['Date'])
    df = df.sort_values('Date')
    df['Day_of_week'] = df['Date'].dt.dayofweek
    df['Week'] = df['Date'].dt.week
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    return df

def add_features_to_df(df: pd.DataFrame) -> pd.DataFrame:
    """ Adds useful features as cols such as the simple moving avg, exponential moving avg, MACD,
    and rsi
    """
    df['Mid'] = (df['Low']+df['High'])/2.0
    df['Day_gain'] = indicators.technical_indicators.day_gain(df)
    df['Ma'] = indicators.technical_indicators.simple_moving_avg(df, center=False)
    df['Ema'] = indicators.technical_indicators.exponential_moving_avg(df, center=False)
    df['Macd_diff'] = indicators.technical_indicators.macd_line(df, center=False) - indicators.technical_indicators.macd_signal(df, center=False)
    df['Rsi'] = indicators.technical_indicators.rsi(df, center=False)
    return df

def convert_date_format(date):
    return dt.datetime.strptime(date, '%Y-%m-%d')

def build_lstm_series(matrix, y_col):
    time_steps = config.time_steps
    dim_0 = matrix.shape[0] - time_steps
    dim_1 = matrix.shape[1]
    x = np.zeros((dim_0, time_steps, dim_1))
    y = np.zeros((dim_0,))
    for i in tqdm(range(dim_0)):
        x[i] = matrix[i:time_steps+i]
        y[i] = matrix[time_steps + i, y_col]
    x = trim_to_batch_size(x)
    y = trim_to_batch_size(y)
    print("shape[0] x = {}, shape[0] y = {}".format(x.shape[0], y.shape[0]))
    return x, y

def trim_to_batch_size(matrix):
    """Trims the matrix.shape[0] to a multiple of the batch size"""
    rows_to_drop = matrix.shape[0] % config.batch_size
    return matrix if rows_to_drop == 0 else matrix[:-rows_to_drop]

def save_metadata(model_name, model, history):
    folder_name = datetime.datetime.now().strftime(config.log_time_format)
    if not os.path.exists('data/metadata/{}'.format(folder_name)):
        os.makedirs('data/metadata/{}'.format(folder_name))
    model.save('data/metadata/{}/{}.h5'.format(folder_name, model_name))
    save_history_json(history, 'data/metadata/{}/{}.json'.format(folder_name, model_name))

def save_history_json(history, path):
    with open(path, 'w') as f:
        json.dump(history.history, f)

def get_stock_data(symbol: str, start='1/1/1800', end='1/1/2100', function='TIME_SERIES_DAILY') -> pd.DataFrame:
	# Load key settings
	with open('config/keys.json') as f:
		keys = json.load(f)
	ts = AVTimeSeriesReader(symbols=symbol, start=start, end=end, function=function, api_key=keys['alpha_vantage_key'])
	df = pd.DataFrame(ts.read())
	ts.close()
	df.index = pd.to_datetime(df.index)
	return df
