import pandas as pd
import numpy as np
import math
import config

def day_gain(df: pd.DataFrame, price_col=config.label_name, window_size=1):
    past_df = pd.DataFrame()
    past_df[price_col] = df[price_col].shift(window_size)
    return (df[price_col] - past_df[price_col]) / past_df[price_col] * 100

def get_weekly_max_gain(current: float, next: float):
    """TODO: Add weekly max gain"""
    return

def simple_moving_avg(df: pd.DataFrame, price_col=config.label_name, window_size=15, center=True):
    return df[price_col].rolling(window=window_size, center=center).mean()

def exponential_moving_avg(df: pd.DataFrame, price_col=config.label_name, window_size=15, center=True):
    if center == True:
        ema_df = df[price_col].shift(int(window_size/2)).ewm(span=window_size).mean()
        return _remove_trailing_data(ema_df, window_size)
    else:
        ema_df = pd.Series.ewm(df[price_col], span=window_size).mean()
        return ema_df

def macd_line(df: pd.DataFrame, ema1_window_size=12, ema2_window_size=26, center=True):
    macd_line_df = exponential_moving_avg(df, window_size=ema1_window_size, center=center) - exponential_moving_avg(df, window_size=ema2_window_size, center=center)
    return macd_line_df

def macd_signal(df: pd.DataFrame, price_col=config.label_name, window_size=9, ema1_window_size=12, ema2_window_size=26, center=True):
    macd_line_df = pd.DataFrame()
    macd_line_df[price_col] = macd_line(df, ema1_window_size=ema1_window_size, ema2_window_size=ema2_window_size, center=center)
    return exponential_moving_avg(macd_line_df, window_size=window_size, center=center)

def rsi(df: pd.DataFrame, price_col=config.label_name, window_size=15, center = True):
    delta, upward, downward, upward_ema, downward_ema = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    delta = df[price_col].diff()
    delta = delta[1:]
    upward[price_col], downward[price_col] = delta, delta
    upward[price_col][upward[price_col] < 0] = 0
    downward[price_col][downward[price_col] > 0] = 0
    upward_ema[price_col] = exponential_moving_avg(upward, window_size=window_size, center=center)
    downward_ema[price_col] = exponential_moving_avg(downward.abs(), window_size=window_size, center=center)
    return 100 - 100/(1+upward_ema[price_col]/downward_ema[price_col])

def stochastic_oscillator_k(df: pd.DataFrame, price_col=config.label_name, window_size=15):
    lowest_low_period = df[price_col].rolling(window=window_size, center=True).min()
    highest_high_period = df[price_col].rolling(window=window_size, center=True).max()
    return (df[price_col] - lowest_low_period)/(highest_high_period - lowest_low_period)*100

def stochastic_oscillator_d(df: pd.DataFrame, k_column, window_size=3):
    return simple_moving_avg(df, k_column, window_size)

def cci(df: pd.DataFrame, close=config.label_name, high='High', low='Low', window_size=15):
    typical_price = pd.DataFrame()
    typical_price[close] = _typical_price(df, close, high, low)
    mean_deviation = typical_price[close].rolling(window= window_size, center=True).apply(_mad, raw=True)
    return (typical_price[close] - simple_moving_avg(df, close, window_size)) \
            / (0.015*mean_deviation)

def money_flow(df: pd.DataFrame, close=config.label_name, high='High', low='Low', volume='Volume'):
    typical_price = pd.DataFrame()
    typical_price[close] = _typical_price(df, close, high, low)
    return typical_price[close] * df[volume]

def _mad(x):
    return np.fabs(x - x.mean()).mean()

def _typical_price(df: pd.DataFrame, close, high, low):
    return (df[close] + df[high] + df[low])/3

def _remove_trailing_data(df: pd.DataFrame, window_size):
    return df.shift(-int(window_size/2)*2).shift(int(window_size/2))

def create_indicator_columns(df: pd.DataFrame, center=False) -> pd.DataFrame:
    """Create columns for indicators with positive and negative signals
    
    Examples of **Positive conditions**:
    1. Moving average - previous moving average is lower
    2. EMA - EMA 10 is higher than EMA 20
    3. MACD - MACD Line higher than MACD signal
    4. RSI - RSI is above 30 but below 70
    5. Stochastic Oscillator - %K less than %D and within 20 - 80
    6. CCI - below -100
    """
    df['MA'] = simple_moving_avg(df, center=center)
    df['+MA'] = df['MA'][df['MA'].diff() > 0]
    df['-MA'] = df['MA'][df['MA'].diff() < 0]
    df['EMA 10'] = exponential_moving_avg(df, window_size=10, center=center)
    df['EMA 20'] = exponential_moving_avg(df, window_size=20, center=center)
    df['+EMA 10'] = df['EMA 10'][df['EMA 10'] > df['EMA 20']]
    df['-EMA 10'] = df['EMA 10'][df['EMA 10'] < df['EMA 20']]
    df['MACD Line'] = macd_line(df, center=center)
    df['MACD Signal'] = macd_signal(df, center=center)
    df['+MACD Line'] = df['MACD Line'][df['MACD Line'] > df['MACD Signal']]
    df['-MACD Line'] = df['MACD Line'][df['MACD Line'] < df['MACD Signal']]
    # df['RSI'] = rsi(df)
    # df['+RSI'] = df['RSI'][df['RSI'] < 30]
    # df['-RSI'] = df['RSI'][df['RSI'] > 70]
    # df['Stochastic Oscillator %K'] = stochastic_oscillator_k(df)
    # df['Stochastic Oscillator %D'] = stochastic_oscillator_d(df, 'Stochastic Oscillator %K')
    # df['+Stochastic Oscillator'] = df['Stochastic Oscillator %K'][np.logical_or(df['Stochastic Oscillator %K'] < 20, np.logical_and(np.logical_not(df['Stochastic Oscillator %K'] > 80), df['Stochastic Oscillator %K'] > df['Stochastic Oscillator %D']))]
    # df['-Stochastic Oscillator'] = df['Stochastic Oscillator %K'][np.logical_or(df['Stochastic Oscillator %K'] > 80, np.logical_and(np.logical_not(df['Stochastic Oscillator %K'] < 20), df['Stochastic Oscillator %K'] < df['Stochastic Oscillator %D']))]
    # df['CCI'] = cci(df)
    # df['+CCI'] = df['CCI'][df['CCI'] < -100]
    # df['-CCI'] = df['CCI'][df['CCI'] > 100]
    return df

def get_action_periods(df: pd.DataFrame, indicator_name: str) -> dict:
    """Gets buy and sell periods according to indicator

    Returns a dictionary containing:
    1. 'buy_periods'
    2. 'sell_periods'
    """
    buy_periods = []
    sell_periods = []
    i = 1
    df_length = len(df)
    while(i < df_length):
        if math.isnan(df[indicator_name].iloc[i-1]) or math.isnan(df[indicator_name].iloc[i]):
            pass
        elif df[indicator_name].iloc[i] > df[indicator_name].iloc[i-1]:
            start = df.index[i]
            while(i < df_length and _is_market_action_condition(df, indicator_name, i, 'buy')):
                i+=1
            end = df.index[i-1]
            if (start < end):
                buy_periods.append([start, end])
        elif df[indicator_name].iloc[i] < df[indicator_name].iloc[i-1]:
            start = df.index[i]
            while(i < df_length and _is_market_action_condition(df, indicator_name, i, 'sell')):
                i+=1
            end = df.index[i-1]
            if (start < end):
                sell_periods.append([start, end])
        i+=1
    return {'buy_periods': buy_periods, 'sell_periods': sell_periods}

def _is_market_action_condition(df: pd.DataFrame, indicator_name: str, i: int, action: str) -> bool:
    """Checks if the market action is correct for row i
    """
    if action == 'buy':
        if indicator_name == 'MA':
            return df['MA'].iloc[i] > df['MA'].iloc[i - 1]
        elif indicator_name == 'EMA 10':
            return df['EMA 10'].iloc[i] > df['EMA 20'].iloc[i]
        elif indicator_name == 'MACD Line':
            return df['MACD Line'].iloc[i] > df['MACD Signal'].iloc[i]
        elif indicator_name == 'RSI':
            return df['RSI'].iloc[i] < 30
        elif indicator_name == 'Stochastic Oscillator %K':
            return df['Stochastic Oscillator %K'].iloc[i] < 20 or ((not df['Stochastic Oscillator %K'].iloc[i] > 80) and df['Stochastic Oscillator %K'].iloc[i] > df['Stochastic Oscillator %D'].iloc[i])
        elif indicator_name == 'CCI':
            return df['CCI'].iloc[i] < -100
        elif indicator_name == 'Money Flow':
            return False
    elif action == 'sell':
        if indicator_name == 'MA':
            return df['MA'].iloc[i] < df['MA'].iloc[i - 1]
        elif indicator_name == 'EMA 10':
            return df['EMA 10'].iloc[i] < df['EMA 20'].iloc[i]
        elif indicator_name == 'MACD Line':
            return df['MACD Line'].iloc[i] < df['MACD Signal'].iloc[i]
        elif indicator_name == 'RSI':
            return df['RSI'].iloc[i] > 70
        elif indicator_name == 'Stochastic Oscillator %K':
            return df['Stochastic Oscillator %K'].iloc[i] >80 or ((not df['Stochastic Oscillator %K'].iloc[i] < 20) and df['Stochastic Oscillator %K'].iloc[i] < df['Stochastic Oscillator %D'].iloc[i])
        elif indicator_name == 'CCI':
            return df['CCI'].iloc[i] > 100
        elif indicator_name == 'Money Flow':
            return False
    else:
        raise Exception("Please check action parameter. Wrong input.")
