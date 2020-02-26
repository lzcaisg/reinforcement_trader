import indicators
import util
import pandas as pd
import math
import calendar
from copy import deepcopy

def get_trend_list(stocks:list, df_list: list, start='1/1/2016', end='31/12/2018'):
    select_indicators = ['MACD Line']
    action_periods_dict = {}

    for i,stock in enumerate(stocks):
        df = df_list[i]
        df = df.loc[(df['Date'] > start) & (df['Date'] <= end)]
        df = indicators.create_indicator_columns(df)
        indicator_dict = {}

        for indicator in select_indicators:
            indicator_dict[indicator] = indicators.get_action_periods(df, indicator)

        # util.plot_market_trends(df, indicator_dict, stock)
        action_periods_dict[stock] = indicator_dict
    buy_trend_set = set()
    sell_trend_set = set()
    for i,symbol in enumerate(action_periods_dict):
        df = df_list[i]
        for period in action_periods_dict[symbol]['MACD Line']['buy_periods']:
            buy_trend_set.add(df['Date'].iloc[period[0]])
            buy_trend_set.add(df['Date'].iloc[period[1]])
        for period in action_periods_dict[symbol]['MACD Line']['sell_periods']:
            sell_trend_set.add(df['Date'].iloc[period[0]])
            sell_trend_set.add(df['Date'].iloc[period[1]])

    # Join 2 sets together, treating each start and end period as a period of action
    trend_list = list(buy_trend_set.union(sell_trend_set))
    temp_trend_list = []
    for date in trend_list:
        df1 = df_list[0][df_list[0]['Date'] == date]
        df2 = df_list[1][df_list[1]['Date'] == date]
        df3 = df_list[2][df_list[2]['Date'] == date]
        if not (df1.empty or df2.empty or df3.empty):
            temp_trend_list.append(date)
    trend_list = sorted(temp_trend_list)
    return trend_list

def cal_portfolio_comp_fitness(asset_list, base_rates, original_portfolio_comp, df_list, date_range, trend_list, cvar_period, mc_period, sp_period, c1, c2, thres, fitness=[]):
    """Calculates the portfolio comp at each change and updates fitness values. Returns a boolean changes list
    """
    change_list = []
    i = 0
    new_portfolio_comp = deepcopy(original_portfolio_comp)
    last_trade_date=[date_range[0]]
    for date in date_range:
        high_risk_date = df_list[0][df_list[0]['Date'] == date]
        med_risk_date = df_list[1][df_list[1]['Date'] == date]
        low_risk_date = df_list[2][df_list[2]['Date'] == date]
        if not (high_risk_date.empty or med_risk_date.empty or low_risk_date.empty):
            if date in trend_list:
                # print('Reallocating at {}'.format(date))
                # Hack for multiple base rates
                if len(base_rates) == 3:
                    new_portfolio_comp = util.get_portfolio_comp(original_portfolio_comp, df_list, base_rates, date,
                        cvar_period, mc_period, sp_period, c1, c2)
                else:
                    new_portfolio_comp = util.get_portfolio_comp(original_portfolio_comp, df_list, base_rates[i], date,
                        cvar_period, mc_period, sp_period, c1, c2)
                    i += 1
    # With commission
                change = cal_nav(date, new_portfolio_comp, df_list, asset_list, last_trade_date, 
                    original_portfolio_comp=original_portfolio_comp, thres=thres)
    # # Without commission
    #             change = cal_nav(date, new_portfolio_comp, df_list, asset_list, last_trade_date)
    # #########################
                original_portfolio_comp = new_portfolio_comp
                change_list.append(change)
            else:
                change_list.append((False, 0, date))
    asset_list = cal_fitness_with_nav(df_list, asset_list, last_trade_date[-1], date_range[-1], fitness)
    return change_list, asset_list, [original_portfolio_comp[0], original_portfolio_comp[1], original_portfolio_comp[2]]

def cal_nav(date, new_portfolio_comp, df_list, asset_list, last_trade_date: list, original_portfolio_comp=[], thres=0, commisson_rate=1.0/800):
    """Updates asset list with calculated new assets. Returns change_list of (True, asset_list, date) or (False, 0, date)
    """
    # without commission
    if thres == 0:
        for i in range(len(new_portfolio_comp)):
            # Update asset values
            previous_close_price = df_list[i][df_list[i]['Date'] == last_trade_date[-1]]['Close'].values[0]
            current_close_price = df_list[i][df_list[i]['Date'] == date]['Close'].values[0]
            asset_list[i] = asset_list[i] * current_close_price / previous_close_price
        for i, composition in enumerate(new_portfolio_comp):
            asset_list[i] = sum(asset_list) * composition
        last_trade_date.append(date)
        new_asset_list = deepcopy(asset_list)
        return (True, new_asset_list, date)
    # with commission
    else:
        percent_change = []
        total_change = 0
        for i in range(len(original_portfolio_comp)):
            change = new_portfolio_comp[i]-original_portfolio_comp[i]
            percent_change.append(change)
            total_change+=abs(change)
        if total_change > thres:
            # print('Portfolio nav changed from {}'.format(asset_list))
            for i in range(len(new_portfolio_comp)):
                # Update asset values
                previous_close_price = df_list[i][df_list[i]['Date'] == last_trade_date[-1]]['Close'].values[0]
                current_close_price = df_list[i][df_list[i]['Date'] == date]['Close'].values[0]
                asset_list[i] = asset_list[i] * current_close_price / previous_close_price

            total_assets = sum(asset_list)
            for i in range(len(new_portfolio_comp)):
                amount_change = new_portfolio_comp[i] * total_assets - asset_list[i]
                if amount_change <= 0:
                    asset_list[i] = asset_list[i] + amount_change
                else:
                    asset_list[i] = asset_list[i] + amount_change * (1 - commisson_rate)**2
            last_trade_date.append(date)
            # print('to {} at {} \n'.format(asset_list, date))
            new_asset_list = deepcopy(asset_list)
            return (True, new_asset_list, date)
    return (False, 0, date)

def cal_fitness_with_nav(df_list, asset_list, last_trade_date, last_date, fitness=[]):
    """Update final asset value and update fitness value if provided
    """
    cvar = 0
    tmp_asset_list = deepcopy(asset_list)
    for i in range(len(df_list)):
        previous_close_price = df_list[i][df_list[i]['Date'] == last_trade_date]['Close'].values[0]
        current_close_price = df_list[i][df_list[i]['Date'] == last_date]['Close'].values[0]
        tmp_asset_list[i] = tmp_asset_list[i] * current_close_price / previous_close_price
    asset_value = sum(tmp_asset_list)
    # print('Portfolio asset value = {}'.format(asset_value))
    for i in range(len(df_list)):
        composition =  tmp_asset_list[i]/asset_value
        cvar_value = util.cvar_percent(df_list[i], len(df_list[i])-1, len(df_list[i])-1) * composition
        cvar += abs(cvar_value)
    # print('Final cvar: {}'.format(cvar))
    fitness_value = asset_value / cvar
    if math.isnan(fitness_value):
        fitness_value = 0
    fitness.append(fitness_value)
    return tmp_asset_list
    # print('Number of trades done = {}'.format(len(last_trade_date)))

def cal_fitness_with_quarterly_returns(daily_df, fitness, price_col='Close'):
    quarterly_dict = {'start_period': [], 'end_period': [], 'quarterly_return': []}
    start_year = daily_df['Date'].iloc[0].year
    end_year = daily_df['Date'].iloc[-1].year
    for year in range(start_year, end_year + 1):
        for quarter_start in range(1,13,3):
            q_start = '{}-{}-1'.format(year, quarter_start)
            q_end = '{}-{}-{}'.format(year, quarter_start + 2, calendar.monthrange(year, quarter_start + 2)[1])
            temp_df = daily_df[(daily_df['Date'] >= q_start) & (daily_df['Date'] <= q_end)]
            # Remove empty data
            temp_df = temp_df[temp_df[price_col] != 0]
            if len(temp_df.index) > 2:
                quarter_start_close = temp_df[price_col].iloc[0]
                quarter_end_close = temp_df[price_col].iloc[-1]
                quarter_return = (quarter_end_close - quarter_start_close) / quarter_start_close * 100
                quarterly_dict['start_period'].append(temp_df['Date'].iloc[0].date())
                quarterly_dict['end_period'].append(temp_df['Date'].iloc[-1].date())
                quarterly_dict['quarterly_return'].append(quarter_return)
    quarterly_df = pd.DataFrame(quarterly_dict)
    fitness_value = sum(quarterly_df['quarterly_return'].values)
    if math.isnan(fitness_value):
        fitness_value = 0
    fitness.append(fitness_value)
    return quarterly_df

def cal_portfolio_changed_nav(asset_list, base_rates, original_portfolio_comp, df_list, date_range, trend_list, cvar_period, mc_period, sp_period, c1, c2, thres, fitness=[]):
    """Calculates the portfolio comp at each change and returns the changed (not average) nav.
    """
    i = 0
    old_asset_sum = 0
    new_portfolio_comp = deepcopy(original_portfolio_comp)
    last_trade_date=[date_range[0]]
    date = date_range[0]
    # print('Reallocating at {}'.format(date))
    # Hack for multiple base rates
    if len(base_rates) == 3:
        new_portfolio_comp = util.get_portfolio_comp(original_portfolio_comp, df_list, base_rates, date,
            cvar_period, mc_period, sp_period, c1, c2)
    else:
        new_portfolio_comp = util.get_portfolio_comp(original_portfolio_comp, df_list, base_rates[i], date,
            cvar_period, mc_period, sp_period, c1, c2)
    # With commission
    cal_nav(date, new_portfolio_comp, df_list, asset_list, last_trade_date, 
        original_portfolio_comp=original_portfolio_comp, thres=thres)
    # # Without commission
    # cal_nav(date, new_portfolio_comp, df_list, asset_list, last_trade_date)
    # #########################
    old_asset_sum = sum(asset_list)
    original_portfolio_comp = new_portfolio_comp

    temp_asset_list = cal_fitness_with_nav(df_list, asset_list, date_range[-1], date)
    new_asset_sum = sum(temp_asset_list)
    changed_asset_sum = new_asset_sum - old_asset_sum
    return changed_asset_sum, asset_list, [original_portfolio_comp[0], original_portfolio_comp[1], original_portfolio_comp[2]]
