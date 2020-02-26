import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
from copy import deepcopy
from tqdm import tqdm
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import datetime
import sys
import math
import util
import config
import indicators

weight_decay_beta = float('10e-9')
run_set = ['goldman', 'index', '^BVSP', '^TWII', '^IXIC', 'index_sampled']
stocks = ['^BVSP', '^TWII', '^IXIC']
# stocks = ['GGSIX', 'GOIIX', 'GIPIX']
path = "data/rl/rl_model_1"
choose_set_num = 1

price_period = 20
risk_level = 1

load_model = False
save_rl_data = True
save_passive = True
save_algo_data = True
df_list, date_range, trend_list, _ = util.get_algo_dataset(choose_set_num)
max_ep_length = len(trend_list)
action_taken_list = []

batch_size = 32
update_freq = 10
gamma = .99
start_e = 1
end_e = 0.1
annealing_steps = 5000
num_episodes = 350
pre_train_steps = 84400 #160000
max_epLength = len(trend_list) - 1
h_size = 100
tau = 0.0005

num_actions = 4
state_dimension = 5

df_list, date_range, trend_list, _ = util.get_algo_dataset(choose_set_num)
max_ep_length = len(trend_list)
action_taken_list = []

#Set the rate of random action decrease.
e_rate = start_e
step_drop = (start_e - end_e)/annealing_steps

# #BVSP
# stock_pred_model = tf.keras.models.load_model(f'data/rl/{run_set[choose_set_num]}/stock_pred.hdf5')
# scaler = MinMaxScaler(feature_range=(0,1))

class Qnetwork():
    def __init__(self, H):
        sum_regularization=0
        self.x = tf.compat.v1.placeholder(tf.float32, [1, state_dimension])
        self.W0 = tf.Variable(tf.random.uniform([state_dimension, H], 0, 1))
        self.b0 = tf.Variable(tf.constant(0.1, shape=[H]))

        self.y_hidden = tf.nn.relu(tf.matmul(self.x, self.W0) + self.b0)
        sum_regularization += weight_decay_beta * tf.nn.l2_loss(self.W0)

        self.W1 = tf.Variable(tf.random.uniform([H, num_actions], 0, 1))
        self.b1 = tf.Variable(tf.constant(0.1, shape=[num_actions]))
        sum_regularization += weight_decay_beta * tf.nn.l2_loss(self.W1)
        # q out
        self.q_values = tf.matmul(self.y_hidden, self.W1) + self.b1
        # predict
        self.best_action = tf.argmax(self.q_values, 1)

        # next q
        self.target = tf.compat.v1.placeholder(tf.float32, [1 ,num_actions])
        self.loss = tf.reduce_sum(tf.square(self.target - self.q_values) + sum_regularization)
        # self.update = tf.compat.v1.train.AdamOptimizer(learning_rate=0.00005).minimize(self.loss)
        self.update = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss)
        # TODO: try higher lr or dropout
        # self.update = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.00005).minimize(self.loss)

def norm_state(state):
    temp = deepcopy(state)
    return np.reshape(np.hstack(temp), (1, state_dimension))

def process_action(action, portfolio_composition):
    new_portfolio_composition = deepcopy(portfolio_composition)
    ####################### Full switch ##############################
    # high risk up, med risk up
    if action == 0:
        for _ in range(3):
            new_portfolio_composition[0] = 0.8
            new_portfolio_composition[1] = 0.1
            new_portfolio_composition[2] = 0.1
    elif action ==1:
        for _ in range(3):
            new_portfolio_composition[0] = 0.1
            new_portfolio_composition[1] = 0.8
            new_portfolio_composition[2] = 0.1
    elif action == 2:
        for _ in range(3):
            new_portfolio_composition[0] = 0.45
            new_portfolio_composition[1] = 0.45
            new_portfolio_composition[2] = 0.1
    elif action == 3:
        for _ in range(3):
            new_portfolio_composition[0] = 0.1
            new_portfolio_composition[1] = 0.1
            new_portfolio_composition[2] = 0.8
    ###############################################################
    ####################### Gradual ##############################
    # # high risk up, med risk up
    # if action == 0:
    #     for _ in range(3):
    #         # low risk base rate enough, L -> H
    #         if new_portfolio_composition[2] - 0.1 >= 0.1:
    #             new_portfolio_composition[0] += 0.1
    #             new_portfolio_composition[2] -= 0.1
    #         # med risk base rate enough, M -> H
    #         if new_portfolio_composition[1] - 0.1 >= 0.1:
    #             new_portfolio_composition[0] += 0.1
    #             new_portfolio_composition[1] -= 0.1
    # elif action ==1:
    #     for _ in range(3):
    #         # high risk base rate enough, H -> M
    #         if new_portfolio_composition[0] - 0.1 >= 0.1:
    #             new_portfolio_composition[1] += 0.1
    #             new_portfolio_composition[0] -= 0.1
    #         # low risk base rate enough, L -> M
    #         if new_portfolio_composition[2] - 0.1 >= 0.1:
    #             new_portfolio_composition[1] += 0.1
    #             new_portfolio_composition[2] -= 0.1
    # elif action == 2:
    #     for _ in range(3):
    #         # low risk base rate enough, L -> H
    #         if new_portfolio_composition[0] - 0.1 >= 0.1:
    #             new_portfolio_composition[2] += 0.1
    #             new_portfolio_composition[0] -= 0.1
    #         # low risk base rate enough, L -> M
    #         if new_portfolio_composition[1] - 0.1 >= 0.1:
    #             new_portfolio_composition[2] += 0.1
    #             new_portfolio_composition[1] -= 0.1
    # elif action == 3:
    #     for _ in range(3):
    #         # high risk base rate enough, H -> L
    #         if new_portfolio_composition[0] - 0.1 >= 0.1:
    #             new_portfolio_composition[2] += 0.1
    #             new_portfolio_composition[0] -= 0.1
    #         # med risk base rate enough, M -> L
    #         if new_portfolio_composition[1] - 0.1 >= 0.1:
    #             new_portfolio_composition[2] += 0.1
    #             new_portfolio_composition[1] -= 0.1
    ################################################
    return new_portfolio_composition

def get_next_state(current_index, trend_list, date_range, df_list):
    date = trend_list[current_index+1]
    date_idx = [i for i, cur_date in enumerate(date_range) if date == cur_date][0]
    state_ = ()

    for i in range(2):
        # Check price data for state. Get the price_period num of days price before period
        price_list = []
        if date_idx - price_period >= 0:
            price_dates = date_range[date_idx-price_period:date_idx]
        else:
            price_dates = date_range[0:date_idx]
            for _ in range(price_period - date_idx):
                price_list.append(0)
    
        for date in price_dates:
            price_list.append(df_list[i][df_list[i]['Date'] == date]['Close'].values[0])

        df = pd.DataFrame({'Close': price_list})
        df['EMA'] = indicators.exponential_moving_avg(df, window_size=6, center=False)
        df['MACD_Line'] = indicators.macd_line(df, ema1_window_size=3, ema2_window_size=6, center=False)
        df['MACD_Signal'] = indicators.macd_signal(df, window_size=6, ema1_window_size=3, ema2_window_size=6, center=False)
        # df['Rsi'] = indicators.rsi(df, window_size=10, center=False)
        # Predict using lstm model and get centered indicators
        if i == 0:
            df = get_predicted_indicator_df(df, price_list, scaler, high_risk_pred_model)
        elif i == 1:
            df = get_predicted_indicator_df(df, price_list, scaler, med_risk_pred_model)
        ema_price = util.z_score_normalization(df.iloc[-1]['EMA'], df['EMA'].tolist())
        macd_line = df['MACD_Line']
        macd_signal = df['MACD_Signal']
        macd = [macd_line.iloc[i] - macd_signal.iloc[i] for i in range(len(macd_line))]
        macd = util.scale(macd[-1], macd)
        if (math.isnan(ema_price) or math.isnan(macd)):
            print(f'nan encountered: ema = {ema_price}, macd = {macd}')

        state_ += (ema_price, macd)

    if current_index == -1:
        last_date_delta = 0
    else:
        last_date_delta = (trend_list[current_index+1] - trend_list[current_index]).days

    state_ += (last_date_delta,)
    return state_

def get_predicted_indicator_df(df, price_list, scaler, model):
    scaled_df = scaler.fit_transform(df)
    temp_data1 = []
    # predict 3 days after. There must be a better way!
    for j in range(3):
        temp_data2 =[]
        # lookback period of 7
        for k in range(7):
            temp_data3 = scaled_df[-j-k-1].tolist()[1:]
            temp_data2.append(temp_data3)
        temp_data1.append(temp_data2)
    temp_data1 = np.array(temp_data1)

    prediction = model.predict(temp_data1)
    temp_data = np.concatenate((prediction, temp_data1[:,0,:]), axis=1)
    pred_close = scaler.inverse_transform(temp_data)[:, 0]
    # print(pred_close, price_list[-1])
    for close in pred_close:
        price_list.append(close)
    df = pd.DataFrame({'Close': price_list})
    df['EMA'] = indicators.exponential_moving_avg(df, window_size=6, center=True)
    df['MACD_Line'] = indicators.macd_line(df, ema1_window_size=3, ema2_window_size=6, center=True)
    df['MACD_Signal'] = indicators.macd_signal(df, window_size=6, ema1_window_size=3, ema2_window_size=6, center=True)
    # print(df.iloc[3:-3])
    return df.iloc[3:-3]

def get_reward(asset_list, action, current_index, trend_list,  date_range, portfolio_composition, df_list):
    new_asset_list = deepcopy(asset_list)
    reward_period = 10
    commisson_rate= 1.0/800

    # get reward from reward_period number of days
    date = trend_list[current_index]
    date_idx = [i for i, cur_date in enumerate(date_range) if date == cur_date][0]
    # Check price data for state. Get the price_period num of days price before period
    if date_idx + reward_period < len(date_range):
        reward_date = date_range[date_idx+reward_period]
    else:
        # Not checked in gradual approach. Fix if time allows
        reward_date = date_range[-1]

    passive_asset_sum, _ = get_reward_asset_sum(new_asset_list, portfolio_composition, date, reward_date, commisson_rate)
    # ####### Gradual ##################
    # changed_asset_sum = 0
    # changed_composition_rates = portfolio_composition
    # changed_asset_list = deepcopy(asset_list)
    # for i, cur_date in enumerate(date_range[date_idx:date_idx+reward_period]):
    #     if cur_date != trend_list[current_index + 1] and i < 3:
    #         # Composition rate 1 day after
    #         changed_composition_rates = process_action(action, portfolio_composition)
    #         changed_asset_sum, changed_asset_list = get_reward_asset_sum(changed_asset_list, changed_composition_rates, cur_date, date_range[date_idx+i+1], commisson_rate)
    #         # print('From: ',cur_date,' to ',date_range[date_idx+i+1])
    #     else:
    #         date = cur_date
    #         break
    # if changed_asset_sum - passive_asset_sum == 0 or passive_asset_sum == 0:
    #     new_asset_list, nav_reward = calc_actions_nav(asset_list, portfolio_composition, trend_list, current_index, date_range)
    #     return 0 , changed_composition_rates, new_asset_list
    # new_asset_list, nav_reward = changed_asset_list, changed_asset_sum

    ##########################
    # #### Full swing #############
    changed_composition_rates = process_action(action, portfolio_composition)
    changed_asset_sum, _ = get_reward_asset_sum(new_asset_list, changed_composition_rates, date, reward_date, commisson_rate)

    if changed_asset_sum - passive_asset_sum == 0 or passive_asset_sum == 0:
        new_asset_list, nav_reward = calc_actions_nav(asset_list, portfolio_composition, trend_list, current_index, date_range)
        return 0 , changed_composition_rates, new_asset_list
    new_asset_list, nav_reward = calc_actions_nav(asset_list, portfolio_composition, trend_list, current_index, date_range)
    # ########################

    trend_list_len = len(trend_list)
    # scale to 0.5-1 depending on trend position
    time_scaling_factor = 0.5 * (trend_list_len - current_index) / trend_list_len + 0.5
    # print('Asset difference: ', changed_asset_sum - passive_asset_sum)
    # print('old portfolio: ', portfolio_composition, 'New: ', changed_composition_rates)
    # print('Date: ', date, 'Change:', (changed_asset_sum - passive_asset_sum) / passive_asset_sum  * 100)
    # print('Changed sum: ', changed_asset_sum, 'Passive sum: ', passive_asset_sum)
    reward = (changed_asset_sum - passive_asset_sum) / passive_asset_sum * time_scaling_factor
    # print('Reward: ', reward)
    # print('nav Reward: ', nav_reward/10000000)
    return reward + nav_reward/10000000, changed_composition_rates, new_asset_list

def get_reward_asset_sum(asset_list, changed_composition_rates, current_date, reward_date, commisson_rate):
    temp_asset_list = deepcopy(asset_list)
    # print(temp_asset_list)
    for i in range(3):
        # Update asset values
        previous_close_price = df_list[i][df_list[i]['Date'] == current_date]['Close'].values[0]
        current_close_price = df_list[i][df_list[i]['Date'] == reward_date]['Close'].values[0]
        # print('Prev_close',previous_close_price,'Current_close',current_close_price)
        temp_asset_list[i] = temp_asset_list[i] * current_close_price / previous_close_price
        # print(temp_asset_list[i])

    total_assets = sum(temp_asset_list)
    # print('total assets',total_assets)
    for i in range(3):
        amount_change = changed_composition_rates[i] * total_assets - asset_list[i]
        if amount_change <= 0:
            temp_asset_list[i] = temp_asset_list[i] + amount_change
        else:
            temp_asset_list[i] = temp_asset_list[i] + amount_change * (1 - commisson_rate)**2
    return sum(temp_asset_list), temp_asset_list
    

def get_action_counts(action_taken_list: list) -> list:
    counts = [0] * num_actions
    for action in action_taken_list:
        counts[action] += 1
    return counts

def calc_actions_nav(asset_list, portfolio_composition, trend_list, index, date_range, final_nav=False, commisson_rate= 1.0/800):
    new_asset_list = deepcopy(asset_list)
    if final_nav:
        # Update for end of date
        for j in range(3):
            # Update asset values
            previous_close_price = df_list[j][df_list[j]['Date'] == trend_list[-1]]['Close'].values[0]
            current_close_price = df_list[j][df_list[j]['Date'] == date_range[-1]]['Close'].values[0]
            new_asset_list[j] = new_asset_list[j] * current_close_price / previous_close_price
        return new_asset_list, sum(new_asset_list)

    # start of training at trend_list idx 10
    if index == 10 -1:
        prev_date = date_range[0]
    else:
        prev_date = trend_list[index-1]
    date = trend_list[index]

    # Update asset values by passive market movement
    for j in range(3):
        previous_close_price = df_list[j][df_list[j]['Date'] == prev_date]['Close'].values[0]
        current_close_price = df_list[j][df_list[j]['Date'] == date]['Close'].values[0]
        new_asset_list[j] = new_asset_list[j] * current_close_price / previous_close_price
    total_assets = sum(new_asset_list)

    # Update asset values by portfolio adjustment
    for j in range(3):
        amount_change = portfolio_composition[j] * total_assets - new_asset_list[j]
        # Reduce composition
        if amount_change <= 0:
            new_asset_list[j] = new_asset_list[j] + amount_change
        # Increase composition. Incur buy and sell commission
        else:
            new_asset_list[j] = new_asset_list[j] + amount_change * (1 - commisson_rate)**2

    return new_asset_list, sum(new_asset_list)

def get_action(q_values: list) -> int:
    return np.argmax(q_values)

main_QN = Qnetwork(h_size)
saver = tf.compat.v1.train.Saver()
#Make a path for our model to be saved in.
Path(path).mkdir(parents=True, exist_ok=True)

with tf.compat.v1.Session() as sess:
    if load_model:
        print('Loading model')
        ckpt = tf.train.get_checkpoint_state(path)
        print(ckpt.model_checkpoint_path)
        saver.restore(sess,ckpt.model_checkpoint_path)

        #BVSP
        high_risk_pred_model = tf.keras.models.load_model(f'data/rl/{run_set[choose_set_num]}/stock_pred_^BVSP.hdf5')
        med_risk_pred_model = tf.keras.models.load_model(f'data/rl/{run_set[choose_set_num]}/stock_pred_^TWII.hdf5')
        scaler = MinMaxScaler(feature_range=(0,1))
        
        ep_reward = 0
        start = 10
        end = len(trend_list) -1
        state = get_next_state(start - 1, trend_list, date_range, df_list)
        base_rate_list = []
        portfolio_composition = [0.1, 0.1, 0.1+0.7]
        portfolio_composition_list = []
        action_taken_list=[]
        reward_list=[]
        asset_list = [100000, 100000, 100000]
        # prev_action = ACTION_CLOSE
        for i in range(start, end):
            # print(norm_state(s))
            qv = sess.run(main_QN.q_values, feed_dict={main_QN.x: norm_state(state)})
            # Remove the extra [] for action
            action = get_action(qv)
            # action = action[0]
            action_taken_list.append(action)
            state_ = get_next_state(i, trend_list, date_range, df_list)
            step_reward, portfolio_composition, asset_list = get_reward(asset_list, action, i, trend_list,  date_range, portfolio_composition, df_list)
            reward_list.append(step_reward)
            portfolio_composition_list.append(portfolio_composition)
            state = state_
        nav, asset_list = calc_actions_nav(asset_list, portfolio_composition, trend_list, end, date_range, final_nav=True)
        print(get_action_counts(action_taken_list))
        print(nav)
        action_df = pd.DataFrame({'Date': trend_list[start:-1], 'Reward': reward_list, 'Action': action_taken_list, 'Portfolio_composition': portfolio_composition_list})
        action_df.to_csv(f'data/rl/{run_set[choose_set_num]}/actions_taken.csv')

        nav_daily_dates_list = []
        nav_daily_composition_list = [[], [], []]
        nav_daily_net_list = []
        daily_price_list = []
        commisson_rate = 1.0/800
        asset_list = [100000, 100000, 100000]
        changed = []
        for date in date_range:
            changed.append(date in trend_list[:-1])
        nav_daily_adjust_list = [change for change in changed]
        j = 0
        last_trade_date = date_range[0]
        reward_period = 10
        for k, date in enumerate(date_range):
            # Generate daily NAV value for visualisation
            current_nav_list = []
            if date in trend_list[start:-1]:
                # Update asset composition
                for i in range(3):
                    previous_close_price = df_list[i][df_list[i]['Date'] == last_trade_date]['Close'].values[0]
                    current_close_price = df_list[i][df_list[i]['Date'] == date]['Close'].values[0]
                    asset_list[i] = asset_list[i] * current_close_price / previous_close_price
                total_assets = sum(asset_list)

                # ##############Gradual ############
                # # Not working
                # action = action_taken_list[j]
                # for i, cur_date in enumerate(date_range[k:k+reward_period]):
                #     if cur_date != trend_list[k + 1] and i < 5:
                #         # Composition rate 1 day after
                #         changed_composition_rates = process_action(action, portfolio_composition)
                #         changed_asset_sum, changed_asset_list = get_reward_asset_sum(asset_list, changed_composition_rates, cur_date, date_range[k+i+1], commisson_rate)
                #         for i in range(3):
                #             current_nav_list.append(changed_asset_list[i])
                #         # print('From: ',cur_date,' to ',date_range[date_idx+i+1])
                #     else:
                #         date = cur_date
                #         break
                ###################################
                # Rebalance portfolio
                for i in range(3):
                    amount_change = portfolio_composition_list[j][i] * total_assets - asset_list[i]
                    # Reduce composition
                    if amount_change <= 0:
                        asset_list[i] = asset_list[i] + amount_change
                    # Increase composition. Incur buy and sell commission
                    else:
                        asset_list[i] = asset_list[i] + amount_change * (1 - commisson_rate)**2
                    current_nav_list.append(asset_list[i])
                last_trade_date = date
                j+=1
                ####################################
            else:
                for i in range(3):
                    previous_close_price = df_list[i][df_list[i]['Date'] == last_trade_date]['Close'].values[0]
                    current_close_price = df_list[i][df_list[i]['Date'] == date]['Close'].values[0]
                    current_nav_list.append(asset_list[i] * current_close_price / previous_close_price)

            nav_daily_dates_list.append(date)
            for i in range(3):
                nav_daily_composition_list[i].append(current_nav_list[i])
            daily_price_list.append(sum(current_nav_list)/300000 *100)
            nav_daily_net_list.append(sum(current_nav_list))

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

        for symbol in stocks:
            symbol_cvar = abs(util.cvar_percent(passive_daily_df, len(passive_daily_df)-1, len(passive_daily_df)-1, price_col=symbol))
            print('Stock cvar {}: {}'.format(symbol, symbol_cvar))
            # print('Stock {} cvar difference = {}'.format(symbol, cvar - symbol_cvar))

        path_str = 'data/rl/{}'.format(run_set[choose_set_num])
        path = Path(path_str)
        path.mkdir(parents=True, exist_ok=True)

        if save_passive:
            passive_daily_df.to_csv(f'data/rl/{run_set[choose_set_num]}/passive_daily_nav.csv')
            passive_quarterly_df.to_csv(f'data/rl/{run_set[choose_set_num]}/passive_quarterly_nav_return.csv')
            print('Passive data saved for {}'.format(run_set[choose_set_num]))

        if save_rl_data:
            daily_df.to_csv(f'data/rl/{run_set[choose_set_num]}/daily_nav.csv')
            quarterly_df.to_csv(f'data/rl/{run_set[choose_set_num]}/quarterly_nav_return.csv')
            daily_price_df.to_csv(f'data/rl/{run_set[choose_set_num]}/daily_price.csv')
            print('Data saved for {}'.format(run_set[choose_set_num]))
        sys.exit(0)
    
    sess.run(tf.compat.v1.global_variables_initializer())
    #BVSP
    high_risk_pred_model = tf.keras.models.load_model(f'data/rl/{run_set[choose_set_num]}/stock_pred_^BVSP.hdf5')
    med_risk_pred_model = tf.keras.models.load_model(f'data/rl/{run_set[choose_set_num]}/stock_pred_^TWII.hdf5')
    scaler = MinMaxScaler(feature_range=(0,1))
    total_steps = 0
    reward_list = []
    position_idx = 0
    portfolio_composition = [0.1, 0.1, 0.1+ 0.7]
    ep_reward = 0
    ep_10_reward = 0
    nav = 0
    nav_10_eps = 0
    for j in tqdm(range(num_episodes)):
        print(f'Total Steps taken: {total_steps}')
        # reward_list.append(ep_reward / ((i % 10) + 1))
        if j % 10 == 0:
            print(f"Episode {j}, Total Steps: {total_steps} Average Reward {ep_10_reward / 10}, Average Nav {nav_10_eps / 10}")
            print(f'exploration rate: {e_rate}')
            ep_10_reward = 0
            nav_10_eps = 0
        # episode_buffer = experience_buffer()

        start = 10
        position_idx = start
        state = get_next_state(position_idx-1, trend_list, date_range, df_list)
        portfolio_composition = [0.1, 0.1, 0.8]
        portfolio_composition_list = []
        action_taken_list = []
        reward_list = []
        nav = 0
        ep_reward = 0
        asset_list = [100000,100000,100000]
        while position_idx < max_ep_length - 1:
            action, q_values = sess.run([main_QN.best_action, main_QN.q_values],feed_dict={main_QN.x:norm_state(state)})
            # Remove the extra [] for action
            action = action[0]
            # print(f'Action: {action}, q_values: {q_values}')
            # Explore
            if np.random.rand(1) < e_rate or total_steps < pre_train_steps:
                action = np.random.randint(0, num_actions)

            action_taken_list.append(action)
            step_reward, portfolio_composition, asset_list = get_reward(asset_list, action, position_idx, trend_list, date_range, portfolio_composition, df_list)
            portfolio_composition_list.append(portfolio_composition)
            ep_reward += step_reward
            reward_list.append(step_reward)
            # print(f'Reward for step: {step_reward}')
            # print(state_)
            # Feed new state to obtain new q_value
            state_ = get_next_state(position_idx, trend_list, date_range, df_list)
            q_values_ = sess.run(main_QN.q_values,feed_dict={main_QN.x:norm_state(state_)})
            # print(f'New q_values {q_values_}')
            # Get max q_value
            max_q_value = np.max(q_values_)
            position_idx += 1
            # print(f'Norm state: {norm_state(state)}')

            total_steps += 1
            target_q = q_values
            target_q[0, action] =  step_reward + gamma*max_q_value
            # print(f'Target q: {target_q}')

            _,W1 = sess.run([main_QN.update, main_QN.W1],feed_dict={main_QN.x:norm_state(state), main_QN.target:target_q})
            
            # Calculate profit at end of episode
              
            # Reduce exploration rate
            if total_steps > pre_train_steps:
                if e_rate > end_e:
                    e_rate -= step_drop
                
            state = state_

        asset_list, nav = calc_actions_nav(asset_list, portfolio_composition, trend_list, position_idx, date_range, final_nav=True)
        print(nav)
        print(get_action_counts(action_taken_list))
        nav_10_eps += nav
        ep_10_reward += ep_reward
        print(f'Reward for episode: {ep_reward}')

        # Save every 50 steps from 200 onwards and before end of training
        if (j % 50 == 0 and j >= 200) or j == num_episodes - 1:
            saver.save(sess,path+'/model.cptk')
            print("Saved model")
            # df = pd.DataFrame({'Date': trend_list[start:-1], 'Reward': reward_list, 'Action': action_taken_list, 'Portfolio_composition': portfolio_composition_list})
            # df.to_csv(f'data/rl/{run_set[choose_set_num]}/iteration{j}.csv')

            
