import indicators
import util
import config
import numpy as np
import pandas as pd
import math
import sys

def get_portfolio_comp(current_comp: list, df_list: list, base_rates: list, date: pd.Timestamp, 
    cvar_period=[10,10,10], mc_period=[10,10,10], sp_period=[10,10,10], c1=[0,0,0], c2=[0,0,0]):
    """ df_list should contain ['High Risk', 'Medium Risk', 'Low Risk'] stocks in order
    """
    stocks = df_list
    sum_base_rates = base_rates[0] + base_rates[1] + base_rates[2]
    # print("Total base rate = {:.2f}. Tactical rate = {:.2f}".format(sum_base_rates, 1-sum_base_rates))
    sum_factors = []
    for i, stock in enumerate(stocks):
        # print('{} Stock'.format(stock_class[i]))
        t = stocks[i].index[stocks[i]['Date'] == date].tolist()
        # If any index does not operate on that date, skip reallocation. Percentile need 2 numbers or more, 0 is always nan
        if t == [] or t[0] == 0 or t[0] == 1:
            return current_comp
        else:
            t = t[0]
        # print('Index at {}'.format(t))
        # print('Risk & Market combo = {}'.format(f_risk[i] * f_mc(stock, t)))
        sp = f_sp(stock, t, int(sp_period[i]))
        mr = f_mr(stock, t, int(cvar_period[i]), c2=c2[i])
        mc = f_mc(stock, t, int(mc_period[i]), c1[i])
        # print(f'sp: {sp}, mr: {mr}, mc: {mc}')

        sum_factor = util.modified_tanh(mr * mc) * sp
        sum_factors.append(sum_factor)
    norm_sum_factors = []
    adjustable_comp = 1 - sum_base_rates
    for i in range(len(sum_factors)):
        norm_sum_factors.append(adjustable_comp*util.softmax(sum_factors)[i])
    return [base_rates[i] + norm_sum_factors[i] for i in range(len(base_rates))]

def f_mr(df: pd.DataFrame, t: int, period=10, alpha=0.95, c2=0, price_col='Close'):
	return abs(cvar_percent(df, t, period, alpha, price_col) + c2)

def f_mc(df, t: int, period=10, c1=0):
	mc_df = indicators.macd_line(df, center=False) - indicators.macd_signal(df, center=False)
	if t-period+1 < 0:
		norm_array_df = mc_df.iloc[0:t+1]
	else:
		norm_array_df = mc_df.iloc[t-period+1:t+1]
	norm_mc = util.z_score_normalization(norm_array_df.values[-1], norm_array_df.values.tolist())
	# print("Market Condition = {:.2f}".format(norm_mc))
	return norm_mc + c1

def f_sp(df:pd.DataFrame, t: int, period=10):
    # mean_ema = ema_df.iloc[t-period+1:t+1].mean()
    # print("Swing Potential = {:.2f}".format(mean_ema))
    # t must be bigger than 2 to normalize
    ema_df = indicators.exponential_moving_avg(df, window_size=period, center=False)
    if t-period+2 < 0:
        return util.z_score_normalization(ema_df.iloc[t] ,ema_df.iloc[0:t+1])
    else:
        return util.z_score_normalization(ema_df.iloc[t] ,ema_df.iloc[t-period+1:t+1])
    # return mean_ema

def value_at_risk_percent(df: pd.DataFrame, t: int, period=10, alpha=0.95, price_col='Close'):
    """Calculates the Value at Risk (VaR) of time period
    """
    # t must be bigger than 2 to evaluate percentile
    if t-period+2 < 0:
        var_df = df.iloc[1:t+1]
    else:
        var_df = df.iloc[t-period+1:t+1]
    if price_col=='Close':
        returns_list = var_df['returns'].dropna().values
    else:
        returns_list = indicators.day_gain(var_df, price_col).dropna().values
    if len(returns_list)==0:
        return 0, []
    return np.percentile(returns_list, 100*(1-alpha)), returns_list

def cvar_percent(df: pd.DataFrame, t: int, period=10, alpha=0.95, price_col='Close'):
    """Conditional VaR (CVaR)
    """
    var_percent, returns_list = value_at_risk_percent(df, t, period=period, alpha=alpha, price_col=price_col)
    if len(returns_list)==0:
        return 0
    lower_than_threshold_returns = [returns for returns in returns_list if returns < var_percent]
    return np.nanmean(lower_than_threshold_returns)
