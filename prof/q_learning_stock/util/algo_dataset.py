import numpy as np
import pandas as pd
import indicators
import config
import util
import sys
from pathlib import Path

def get_algo_dataset(choose_set_num: int):
    """run_set = ['goldman', 'index', '^BVSP', '^TWII', '^IXIC', 'index_sampled']
    Returns df_list, date_range, trend_list, stocks
    """
    # Do not change run_set order. The order is hardcoded into below code
    run_set = ['goldman', 'index', '^BVSP', '^TWII', '^IXIC', 'index_sampled']
    choose_set = run_set[choose_set_num]
    df_list = []
    date_range = []
    trend_list = []
    stocks = []

    ### For GS stocks: 'GGSIX', 'GOIIX', 'GIPIX'
    if choose_set == run_set[0]:
        # Must be same order
        stocks = ['GGSIX', 'GOIIX', 'GIPIX']
        folder = ['growth', 'growth_income', 'balanced']
        for i, stock in enumerate(stocks):
            df=pd.read_csv('data/goldman/portfolio/{}/{}.csv'.format(folder[i],stock), usecols=config.column_names, parse_dates=['Date'])
            df = df[df['Close'] > 0].reset_index(drop=True)
            df['returns'] = indicators.day_gain(df, 'Close').dropna()
            df_list.append(df)

        start = '1/1/2016'
        end = '31/12/2018'
        # date_range = df_list[0][(df_list[0]['Date'] >= df_list[1].iloc[0]['Date']) & (df_list[0]['Date'] >= df_list[2].iloc[0]['Date'])]['Date'].tolist()
        date_range = remove_uncommon_dates(df_list)
        trend_list = util.get_trend_list(stocks, df_list, start=start, end=end)

    ### For Index stocks: '^BVSP', '^TWII', '^IXIC'
    elif choose_set == run_set[1]:
        stocks = ['^BVSP', '^TWII', '^IXIC']
        high_risk_df = pd.read_csv('data/indexes/{}.csv'.format('^BVSP'), usecols=config.column_names, parse_dates=['Date'])
        high_risk_df = high_risk_df[high_risk_df['Close'] > 0].reset_index(drop=True)
        high_risk_df['returns'] = indicators.day_gain(high_risk_df, 'Close').dropna()
        med_risk_df = pd.read_csv('data/indexes/{}.csv'.format('^TWII'), usecols=config.column_names, parse_dates=['Date'])
        med_risk_df = med_risk_df[med_risk_df['Close'] > 0].reset_index(drop=True)
        med_risk_df['returns'] = indicators.day_gain(med_risk_df, 'Close').dropna()
        low_risk_df = pd.read_csv('data/indexes/{}.csv'.format('^IXIC'), parse_dates=['Date'])
        # IXIC dates are reversed
        low_risk_df = low_risk_df.reindex(index=low_risk_df.index[::-1])
        low_risk_df = low_risk_df[low_risk_df['Close'] > 0].reset_index(drop=True)
        low_risk_df['returns'] = indicators.day_gain(low_risk_df, 'Close').dropna()

        df_list = [high_risk_df, med_risk_df, low_risk_df]
        start = '1/1/2014'
        end = '31/12/2018'
        # date_range = high_risk_df[(high_risk_df['Date'] >= med_risk_df.iloc[0]['Date']) & (high_risk_df['Date'] >= low_risk_df.iloc[0]['Date'])]['Date'].tolist()
        date_range = remove_uncommon_dates(df_list)
        trend_list = util.get_trend_list(stocks, df_list, start=start, end=end)

    elif choose_set == run_set[2] or choose_set == run_set[3] or choose_set == run_set[4]:
        stocks =[]
        folder =''
        if choose_set == run_set[2]:
            stocks = ['EQTL3.SA', 'ITSA4.SA', 'PETR3.SA']
            folder = '^BVSP'
        elif choose_set==run_set[3]:
            stocks = ['1326.TW', '2882.TW', '3008.TW']
            folder = '^TWII'
        elif choose_set==run_set[4]:
            stocks = ['TSLA', 'IBKC', 'FEYE']
            folder = '^IXIC'
        else:
            print('An error occured in fetching the data for algo stocks.')

        for stock in stocks:
            df=pd.read_csv('data/algo/{}/{}.csv'.format(folder,stock), usecols=config.column_names, parse_dates=['Date'])
            df = df[df['Close'] > 0].reset_index(drop=True)
            df['returns'] = indicators.day_gain(df, 'Close').dropna()
            df_list.append(df)

        start = '1/1/2014'
        end = '31/12/2018'
        
        date_range = remove_uncommon_dates(df_list)
        trend_list = util.get_trend_list(stocks, df_list, start=start, end=end)

    elif choose_set == run_set[5]:
        stocks =['^BVSP', '^TWII', '^IXIC']
        for stock in stocks:
            df=pd.read_csv('data/algo/{}/daily_price.csv'.format(stock), parse_dates=['Date'])
            df = df[df['Close'] > 0].reset_index(drop=True)
            df['returns'] = indicators.day_gain(df, 'Close').dropna()
            df_list.append(df)

        start = '1/1/2014'
        end = '31/12/2018'
        date_range = remove_uncommon_dates(df_list)
        trend_list = util.get_trend_list(stocks, df_list, start=start, end=end)
    return df_list, date_range, trend_list, stocks

def get_algo_results(choose_set_num: int, asset_list, base_rates, portfolio_comp, df_list, date_range, trend_list, cal_avg_nav=False):
    """Returns the change list and final asset value
    """
    change_list = []
    average_asset = 0
    if cal_avg_nav:
        if choose_set_num == 0:
            average_asset, asset_list, portfolio_comp = util.cal_portfolio_changed_nav(asset_list, base_rates, portfolio_comp, df_list, date_range, trend_list, 
                [8.0, 6.0, 12.0], [9.0, 5.0, 9.0], [6.0, 12.0, 6.0], [0.9712034471256101, -1.6709072749507035, -1.0777099909032646], [-3.4145406491989023, -0.18272123074956848, -0.7245604433339186], 0.0816132948369838)
                # [8.0, 8.0, 4.0], [5.0, 6.0, 5.0], [5.0, 2.0, 3.0], [0.22948733470032123, 0.8909251765940478, -0.20656673058505381], [-1.7417846430478365, -0.4628863373977188, 1.5419043896500977], 0.14266550931364091)
                # [21.0, 6.0, 5.0], [2.0, 2.0, 6.0], [27.0, 12.0, 3.0], [3.125115822639779, -2.561089882241202, -1.4940972093691949], [1.2063367792987396, 1.4663555035726752, -0.2846560129041551], 0.1614246940280476)
        elif choose_set_num == 1:
            average_asset, asset_list, portfolio_comp = util.cal_portfolio_changed_nav(asset_list, base_rates, portfolio_comp, df_list, date_range, trend_list, 
                # [8.0, 6.0, 12.0], [9.0, 5.0, 9.0], [6.0, 12.0, 6.0], [0.9712034471256101, -1.6709072749507035, -1.0777099909032646], [-3.4145406491989023, -0.18272123074956848, -0.7245604433339186], 0.0816132948369838)
                # [5.0, 5.0, 6.0], [5.0, 6.0, 6.0], [19.0, 5.0, 8.0], [1.8954915289833882, -1.450482294216655, 1.125418440357023], [-2.3676311336976132, -1.8970317071693157, 0.23699516374694385], 0.046795990258734835)
                [8.0, 14.0, 11.0], [11.0, 11.0, 2.0], [15.0, 10.0, 2.0], [1.363647435463774, 2.716953337278016, -4.324164482875698], [-1.7062595953617727, 2.5105760118208957, -4.060094673509836], 0.07240419552333409)
        elif choose_set_num == 2:
            average_asset, asset_list, portfolio_comp = util.cal_portfolio_changed_nav(asset_list, base_rates, portfolio_comp, df_list, date_range, trend_list,
                # [4.0, 2.0, 4.0], [4.0, 8.0, 9.0], [6.0, 4.0, 7.0], [0.6078976284270344, 1.2577097768694967, 2.0213163271738006], [-2.566918900257593, 2.90468608230902, -1.7097040021899894], 0.07797085783765784)
                [3.0, 3.0, 13.0], [11.0, 5.0, 9.0], [8.0, 4.0, 18.0], [0.06083023158629253, 0.5601483772918827, 1.9569019466459423], [-1.3881334364246258, 2.8163651325079524, 0.9492765355184316], 0.15511606897450375)
        elif choose_set_num == 3:
            average_asset, asset_list, portfolio_comp = util.cal_portfolio_changed_nav(asset_list, base_rates, portfolio_comp, df_list, date_range, trend_list,
                [3.0, 9.0, 4.0], [4.0, 14.0, 3.0], [2.0, 2.0, 16.0], [0.30059198706758106, 1.0952845039110184, 1.8392867588452613], [2.771352403174757, -1.3669589385046343, -2.3406274217770866], 0.17345428438145236)
        elif choose_set_num == 4:
            average_asset, asset_list, portfolio_comp = util.cal_portfolio_changed_nav(asset_list, base_rates, portfolio_comp, df_list, date_range, trend_list,
                # [9.0, 7.0, 9.0], [4.0, 3.0, 7.0], [6.0, 5.0, 15.0], [0.9351583394555885, 1.3754760765507819, 2.348134831028588], [-2.471478593919233, 1.379869639191209, 4.95188889034387], 0.1444277817979811)
                # [8.0, 11.0, 2.0], [6.0, 8.0, 6.0], [7.0, 8.0, 12.0], [1.1255518400058317, -0.36346414388153225, -1.0247284676654485], [-0.6274220138552453, -1.1083765565671055, 0.00449200835519481], 0.13718457807344167)
                [2.0, 5.0, 11.0], [4.0, 2.0, 2.0], [7.0, 5.0, 5.0], [0.2774502065258735, 0.16677941009065034, -0.45385907412444926], [-0.2098008442952385, 1.289022800463935, 2.003346238448586], 0.15779763053682244)
        elif choose_set_num == 5:
            average_asset, asset_list, portfolio_comp = util.cal_portfolio_changed_nav(asset_list, base_rates, portfolio_comp, df_list, date_range, trend_list,
                # [7.0, 12.0, 3.0], [2.0, 7.0, 2.0], [13.0, 3.0, 8.0], [2.522702769828708, -0.5707216899389504, 0.8348229423350395], [-1.7493395408023145, 1.0817636863501934, 0.8232680695157204], 0.1963583867900387)
                [4.0, 6.0, 3.0], [2.0, 4.0, 7.0], [14.0, 2.0, 5.0], [1.3929077534652725, 0.18393055682065484, 2.6440755858307075], [-1.601189152927202, 1.3377505947800103, -1.9787536808104849], 0.13726920065461523)
        else:
            print('ERROR! Wrong choose_set_num')
        return average_asset, asset_list, portfolio_comp
    else:
        if choose_set_num == 0:
            change_list, asset_list, portfolio_comp = util.cal_portfolio_comp_fitness(asset_list, base_rates, portfolio_comp, df_list, date_range, trend_list, 
                [8.0, 6.0, 12.0], [9.0, 5.0, 9.0], [6.0, 12.0, 6.0], [0.9712034471256101, -1.6709072749507035, -1.0777099909032646], [-3.4145406491989023, -0.18272123074956848, -0.7245604433339186], 0.0816132948369838)
                # [8.0, 8.0, 4.0], [5.0, 6.0, 5.0], [5.0, 2.0, 3.0], [0.22948733470032123, 0.8909251765940478, -0.20656673058505381], [-1.7417846430478365, -0.4628863373977188, 1.5419043896500977], 0.14266550931364091)
                # [21.0, 6.0, 5.0], [2.0, 2.0, 6.0], [27.0, 12.0, 3.0], [3.125115822639779, -2.561089882241202, -1.4940972093691949], [1.2063367792987396, 1.4663555035726752, -0.2846560129041551], 0.1614246940280476)
        elif choose_set_num == 1:
            change_list, asset_list, portfolio_comp = util.cal_portfolio_comp_fitness(asset_list, base_rates, portfolio_comp, df_list, date_range, trend_list, 
                # [8.0, 6.0, 12.0], [9.0, 5.0, 9.0], [6.0, 12.0, 6.0], [0.9712034471256101, -1.6709072749507035, -1.0777099909032646], [-3.4145406491989023, -0.18272123074956848, -0.7245604433339186], 0.0816132948369838)
                # [5.0, 5.0, 6.0], [5.0, 6.0, 6.0], [19.0, 5.0, 8.0], [1.8954915289833882, -1.450482294216655, 1.125418440357023], [-2.3676311336976132, -1.8970317071693157, 0.23699516374694385], 0.046795990258734835)
                [8.0, 14.0, 11.0], [11.0, 11.0, 2.0], [15.0, 10.0, 2.0], [1.363647435463774, 2.716953337278016, -4.324164482875698], [-1.7062595953617727, 2.5105760118208957, -4.060094673509836], 0.07240419552333409)
        elif choose_set_num == 2:
            change_list, asset_list, portfolio_comp = util.cal_portfolio_comp_fitness(asset_list, base_rates, portfolio_comp, df_list, date_range, trend_list,
                # [4.0, 2.0, 4.0], [4.0, 8.0, 9.0], [6.0, 4.0, 7.0], [0.6078976284270344, 1.2577097768694967, 2.0213163271738006], [-2.566918900257593, 2.90468608230902, -1.7097040021899894], 0.07797085783765784)
                [3.0, 3.0, 13.0], [11.0, 5.0, 9.0], [8.0, 4.0, 18.0], [0.06083023158629253, 0.5601483772918827, 1.9569019466459423], [-1.3881334364246258, 2.8163651325079524, 0.9492765355184316], 0.15511606897450375)
        elif choose_set_num == 3:
            change_list, asset_list, portfolio_comp = util.cal_portfolio_comp_fitness(asset_list, base_rates, portfolio_comp, df_list, date_range, trend_list,
                [3.0, 9.0, 4.0], [4.0, 14.0, 3.0], [2.0, 2.0, 16.0], [0.30059198706758106, 1.0952845039110184, 1.8392867588452613], [2.771352403174757, -1.3669589385046343, -2.3406274217770866], 0.17345428438145236)
        elif choose_set_num == 4:
            change_list, asset_list, portfolio_comp = util.cal_portfolio_comp_fitness(asset_list, base_rates, portfolio_comp, df_list, date_range, trend_list,
                # [9.0, 7.0, 9.0], [4.0, 3.0, 7.0], [6.0, 5.0, 15.0], [0.9351583394555885, 1.3754760765507819, 2.348134831028588], [-2.471478593919233, 1.379869639191209, 4.95188889034387], 0.1444277817979811)
                # [8.0, 11.0, 2.0], [6.0, 8.0, 6.0], [7.0, 8.0, 12.0], [1.1255518400058317, -0.36346414388153225, -1.0247284676654485], [-0.6274220138552453, -1.1083765565671055, 0.00449200835519481], 0.13718457807344167)
                [2.0, 5.0, 11.0], [4.0, 2.0, 2.0], [7.0, 5.0, 5.0], [0.2774502065258735, 0.16677941009065034, -0.45385907412444926], [-0.2098008442952385, 1.289022800463935, 2.003346238448586], 0.15779763053682244)
        elif choose_set_num == 5:
            change_list, asset_list, portfolio_comp = util.cal_portfolio_comp_fitness(asset_list, base_rates, portfolio_comp, df_list, date_range, trend_list,
                # [7.0, 12.0, 3.0], [2.0, 7.0, 2.0], [13.0, 3.0, 8.0], [2.522702769828708, -0.5707216899389504, 0.8348229423350395], [-1.7493395408023145, 1.0817636863501934, 0.8232680695157204], 0.1963583867900387)
                [4.0, 6.0, 3.0], [2.0, 4.0, 7.0], [14.0, 2.0, 5.0], [1.3929077534652725, 0.18393055682065484, 2.6440755858307075], [-1.601189152927202, 1.3377505947800103, -1.9787536808104849], 0.13726920065461523)
        else:
            print('ERROR! Wrong choose_set_num')
        return change_list, asset_list, portfolio_comp

def gen_algo_data(run_set: list, choose_set_num: int, save_algo_data=False, save_passive=False, save_sub_folder='', is_rl_data=False, base_rates=[], portfolio_comp=[]):
    df_list, date_range, trend_list, stocks = util.get_algo_dataset(choose_set_num)
    # this is an afterthought
    if base_rates == []:
        base_rates = [0.2, 0.2, 0.2]
    if portfolio_comp == []:
        portfolio_comp = [base_rates[i] + [0.4/3, 0.4/3, 0.4/3][i] for i in range(len(base_rates))]
    asset_list = [100000, 100000, 100000]
    change_list = []
    # print('Initial portfolio composition: {}'.format(portfolio_comp))

    change_list,_,_ = util.get_algo_results(choose_set_num, asset_list, base_rates, portfolio_comp, df_list, date_range, trend_list)

    print('Reallocated {} times'.format(len([i for i in change_list if i[0]])))
    # print([i[1] for i in change_list if i[0]])
    nav_daily_dates_list = []
    nav_daily_composition_list = [[], [], []]
    nav_daily_net_list = []
    daily_price_list = []
    asset_list = [100000, 100000, 100000]
    nav_daily_adjust_list = [i[0] for i in change_list]
    j = 0
    last_trade_date = date_range[0]
    for date in date_range:
        # Generate daily NAV value for visualisation
        high_risk_date = df_list[0][df_list[0]['Date'] == date]
        med_risk_date = df_list[1][df_list[1]['Date'] == date]
        low_risk_date = df_list[2][df_list[2]['Date'] == date]
        if not (high_risk_date.empty or med_risk_date.empty or low_risk_date.empty):
            current_nav_list = []
            if not change_list[j][0]:
                for i in range(len(portfolio_comp)):
                    previous_close_price = df_list[i][df_list[i]['Date'] == last_trade_date]['Close'].values[0]
                    current_close_price = df_list[i][df_list[i]['Date'] == date]['Close'].values[0]
                    current_nav_list.append(asset_list[i] * current_close_price / previous_close_price)
            else:
                for i in range(len(portfolio_comp)):
                    asset_list[i] = change_list[j][1][i]
                    current_nav_list.append(asset_list[i])
                last_trade_date = change_list[j][2]
            nav_daily_dates_list.append(date)
            for i in range(len(portfolio_comp)):
                nav_daily_composition_list[i].append(current_nav_list[i])
            daily_price_list.append(sum(current_nav_list)/300000 *100)
            nav_daily_net_list.append(sum(current_nav_list))
            j+=1

    # Note that we are using the Laspeyres Price Index for calculation
    daily_price_df = pd.DataFrame({'Date': nav_daily_dates_list, 'Close': daily_price_list})

    daily_df = pd.DataFrame({'Date': nav_daily_dates_list,\
        stocks[0]: nav_daily_composition_list[0],\
        stocks[1]: nav_daily_composition_list[1],\
        stocks[2]: nav_daily_composition_list[2],\
        'Net': nav_daily_net_list,\
        'Adjusted': nav_daily_adjust_list})

    # Generate quarterly NAV returns for visualisation
    quarterly_df = util.cal_fitness_with_quarterly_returns(daily_df, [], price_col='Net')

    # Generate passive NAV returns for comparison (buy and hold)
    # assets are all 300000 to be able to compare to algo
    asset_list = [300000, 300000, 300000]
    last_date = nav_daily_dates_list[0]
    passive_nav_daily_composition_list = [[],[],[]]
    for date in nav_daily_dates_list:
        for i in range(len(stocks)):
            previous_close_price = df_list[i][df_list[i]['Date'] == last_date]['Close'].values[0]
            current_close_price = df_list[i][df_list[i]['Date'] == date]['Close'].values[0]
            asset_list[i] = asset_list[i] * current_close_price / previous_close_price
            passive_nav_daily_composition_list[i].append(asset_list[i])
        last_date = date

    passive_daily_df = pd.DataFrame({'Date': nav_daily_dates_list,\
        stocks[0]: passive_nav_daily_composition_list[0],\
        stocks[1]: passive_nav_daily_composition_list[1],\
        stocks[2]: passive_nav_daily_composition_list[2]})

    passive_quarterly_df = pd.DataFrame()
    for i in range(len(stocks)):
        if i == 0:
            passive_quarterly_df = util.cal_fitness_with_quarterly_returns(passive_daily_df, [], price_col=stocks[i])
            passive_quarterly_df = passive_quarterly_df.rename(columns={"quarterly_return": stocks[i]})
        else:
            passive_quarterly_df[stocks[i]] = util.cal_fitness_with_quarterly_returns(passive_daily_df, [], price_col=stocks[i])['quarterly_return']
    # print(passive_quarterly_df)

    # Print some quarterly difference statistics
    for symbol in stocks:
        difference = quarterly_df['quarterly_return'].values - passive_quarterly_df[symbol].values
        # print('Stock {}: {}'.format(symbol, difference))
        print('Stock {} total return difference = {}'.format(symbol,sum(difference)))
    # cvar = 0
    # for symbol in stocks:
    #     composition = daily_df[symbol].iloc[-1]/daily_df['Net'].iloc[-1]
    #     cvar_value = util.cvar_percent(daily_df, len(daily_df)-1, len(daily_df)-1, price_col=symbol) * composition
    #     print(cvar_value)
    #     cvar += abs(cvar_value)
    # print('Portfolio cvar = {}'.format(cvar))
    for symbol in stocks:
        symbol_cvar = abs(util.cvar_percent(passive_daily_df, len(passive_daily_df)-1, len(passive_daily_df)-1, price_col=symbol))
        print('Stock cvar {}: {}'.format(symbol, symbol_cvar))
        # print('Stock {} cvar difference = {}'.format(symbol, cvar - symbol_cvar))

    path_str = ''
    if is_rl_data:
        path_str = 'data/rl/{}'.format(run_set[choose_set_num])
        path = Path(path_str)
        path.mkdir(parents=True, exist_ok=True)
    else:
        path_str = 'data/algo/{}'.format(run_set[choose_set_num])
        path = Path(path_str)
        path.mkdir(parents=True, exist_ok=True)

    path = Path(f'{path_str}/{save_sub_folder}')
    path.mkdir(parents=True, exist_ok=True)

    if save_passive:
        passive_daily_df.to_csv(f'{path_str}/{save_sub_folder}passive_daily_nav.csv')
        passive_quarterly_df.to_csv(f'{path_str}/{save_sub_folder}passive_quarterly_nav_return.csv')
        print('Passive data saved for {}'.format(run_set[choose_set_num]))

    if save_algo_data:
        daily_df.to_csv(f'{path_str}/{save_sub_folder}daily_nav.csv')
        quarterly_df.to_csv(f'{path_str}/{save_sub_folder}quarterly_nav_return.csv')
        daily_price_df.to_csv(f'{path_str}/{save_sub_folder}daily_price.csv')
        print('Data saved for {}'.format(run_set[choose_set_num]))

def remove_uncommon_dates(df_list):
    date_range = []
    # temp_date_range = df_list[0][(df_list[0]['Date'] >= df_list[1].iloc[0]['Date']) & (df_list[0]['Date'] >= df_list[2].iloc[0]['Date'])]['Date'].tolist()
    for date in df_list[0]['Date']:
        empty = 0
        for df in df_list:
            temp_df = df[df['Date'] == date]
            if temp_df.empty:
                empty +=1
        if empty == 0:
            date_range.append(date)
    return date_range