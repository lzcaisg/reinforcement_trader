'''
Enables Parameters settings of frequency, currency leakage, start-end, MDD window size
'''

import random
import json
import gym
from gym import spaces
import pandas as pd
import numpy as np


MAX_ACCOUNT_BALANCE = 2147483647
MAX_NUM_SHARES = 2147483647
MAX_SHARE_PRICE = 5000
MAX_OPEN_POSITIONS = 5
MAX_STEPS = 20000
COMMISSION_FEE = 0.00125

INITIAL_ACCOUNT_BALANCE = 300000

DEFAULT_PARAMETER = {
    "trans_freq": 20,
    "have_currency_leakage": True,
    "crisis_detection": False,
    "MDD_window": 20,
    "reward_wait": 10,
    "MDD_threshold": 0.2
}

class RebalancingEnv(gym.Env):

    def __init__(self, df_dict, col_list, env_param, startDate=None, endDate=None, isTraining=True):
        
        super(RebalancingEnv, self).__init__()

        self.have_currency_leakage = env_param['have_currency_leakage']
        self.training = isTraining
        self.col_list = list(col_list)
        if not env_param['have_currency_leakage']:
            self.col_list.remove('Cum FX Change')
        self.reward_wait = env_param['reward_wait']      # reward_wait: Number of days need to wait for determine the reward.
        self.action_freq = env_param['trans_freq']       # Take action for every 7 days.
        self.crisis_detection = env_param['crisis_detection']
        self.MDD_window = env_param['MDD_window']
        self.drawdown_threshold = env_param['MDD_threshold']
        self.df_list = []
        self.market_number = len(df_dict)   # Determine the number of markets
        # 1. Get the intersect dates from different stocks
        tmp_df = df_dict["high"].dropna()
        self.intersect_dates = tmp_df['Date']
        for key in df_dict:
            df = df_dict[key].dropna()      # Remove all NAN in the df
            self.intersect_dates = np.intersect1d(self.intersect_dates, df['Date'])

        # 2. Add only the common dates
        for key in ['high', 'mid', 'low']:
            df = df_dict[key].dropna()
            self.df_list.append(df[df['Date'].isin(self.intersect_dates)].reset_index(drop=True))
        
        self.roughStartDate = startDate
        self.roughEndDate = endDate
        self.start_date = np.min(self.intersect_dates)
        if not self.roughStartDate is None:       # start_date may not in intersect_dates
            self.start_date = max(self.roughStartDate, self.start_date)
        self.start_step = self.df_list[0].index[self.df_list[0]['Date'] >= self.start_date].tolist()[0]
        

        self.end_date = np.max(self.intersect_dates)
        if not self.roughEndDate is None:
            self.end_date = min(self.roughEndDate, self.end_date)
        self.end_step = self.df_list[0].index[self.df_list[0]['Date'] <= self.end_date].tolist()[-1]
        

        # 3. Action Space: [0,0,0,0]->[1,1,1,1]: Four Actions
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0]), high=np.array([1.0, 1.0, 1.0, 1.0]), dtype=np.float16)

        # 4. Observation Space
        
        self.obs_relative_steps = np.array([-1, -2, -3, -4, -5, -10, -15, -20, -40, -100])    # Set the current step to a random point within the data frame
        self.window_size = abs(self.obs_relative_steps[-1])
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(len(self.obs_relative_steps), len(self.col_list)*2), dtype=np.float16)


    def reset(self, startDate=None, endDate=None):
        # Reset the state of the environment to an initial state
        
        '''We set the current step to a random point within the data frame, because it 
        essentially gives our agent’s more unique experiences from the same data set.'''

        '''
        Taking the action in the environment.
        Action: [0,0,0,0]-[1,1,1,1] -> a decision in 0-3
        Action 0: 80% High, 10% Mid, 10% Low;
        Action 1: 10% High, 80% Mid, 10% Low;
        Action 2: 45% High, 45% Mid, 10% Low;
        Action 3: 10% High, 10% Mid, 80% Low.
        '''
        self.action_reference = {
            0:np.array([0.80, 0.10, 0.10]),
            1:np.array([0.10, 0.80, 0.10]),
            2:np.array([0.45, 0.45, 0.10]),
            3:np.array([0.10, 0.10, 0.80])
        }
        
        self.start_date = np.min(self.intersect_dates)
        if not self.roughStartDate is None:       # start_date may not in intersect_dates
            self.start_date = max(self.roughStartDate, self.start_date)
        self.start_step = self.df_list[0].index[self.df_list[0]['Date'] >= self.start_date].tolist()[0]
        
        self.end_date = np.max(self.intersect_dates)
        if not self.roughEndDate is None:
            self.end_date = min(self.roughEndDate, self.end_date)
        self.end_step = self.df_list[0].index[self.df_list[0]['Date'] <= self.end_date].tolist()[-1]

        self.total_net_worth = INITIAL_ACCOUNT_BALANCE
        self.prev_total_net_worth = INITIAL_ACCOUNT_BALANCE
        self.max_net_worth = INITIAL_ACCOUNT_BALANCE
        self.current_step = self.start_step
        self.zero_action_count = 0
        self.prev_buyNhold_balance = 0
        self.finished = False
        self.finished_twice = False
        self.prev_action_step = 0 

        self.net_worth = np.array(
            [INITIAL_ACCOUNT_BALANCE / self.market_number] * self.market_number, 
            dtype=np.float64)

        self.cash = 0
        self.cash_out_trigger = False
        
        self.total_sales_value = np.array([0.0] * self.market_number)
        self.prev_net_worth = self.net_worth
        
        self.prev_action = np.array([0.0,0.0,0.0,0.0])
        self.current_action = self.prev_action

        if self.training:
            days_range = len(self.intersect_dates)
            rand_days = random.randint(self.window_size, days_range - 1)
            self.current_step = rand_days
        else:
            self.current_step = self.start_step

        init_price = [df.loc[self.current_step, "Price"] for df in self.df_list]
        init_price = np.array(init_price, dtype=np.float64)

        self.prev_buyNhold_price = init_price
        self.init_buyNhold_number = (INITIAL_ACCOUNT_BALANCE/self.market_number) / init_price
        self.buyNhold_balance = INITIAL_ACCOUNT_BALANCE

        self.inventory_number = self.init_buyNhold_number
        self.prev_inventory_num = self.inventory_number
        self.cost_basis = init_price
        self.open_for_transaction = True

        return self._next_observation()


    def _next_observation(self):
        '''
        The _next_observation method provices observations in the format of:
        States = ['EMA', 'MACD_diff', 'delta_time', 'RSI', 'Cum FX Change']
        Refer to the report for detailed explanation.
        '''
        # self.current_step is defined in reset method,
        # We assume the current_step is TODAY (BEFORE FINAL), which means we only know infomation till YESTERDAY ENDS.
        # today_date = self.intersect_dates(self.current_step)
        obs_steps = self.current_step+self.obs_relative_steps # obs_relative_steps and obs_steps Is a List!!!   
        
        obs_steps[obs_steps<0] = 0
        high_obs = self.df_list[0][self.col_list].loc[list(obs_steps)]
        mid_obs  = self.df_list[1][self.col_list].loc[list(obs_steps)]
        # low_obs = self.df_list[2][self.col_list].loc[list(obs_steps)]

        obs = pd.concat([high_obs, mid_obs], axis=1, sort=False)
        # obs = pd.concat([high_obs, mid_obs, low_obs], axis=1, sort=False)
        # obs.columns = [tmp+'_h' for tmp in self.col_list] + [tmp+'_m' for tmp in self.col_list] + [tmp+'_l' for tmp in self.col_list]
        obs.columns = [tmp+'_h' for tmp in self.col_list] + [tmp+'_m' for tmp in self.col_list]
        obs.reset_index(inplace=True, drop=True)
        # obs[['EMA_h', 'EMA_m']] /= obs[['EMA_h', 'EMA_m']].iloc[0]
        theSum = abs(obs.values).sum()
        if theSum == np.inf:
            print(obs)
        self.obs = obs

        return self.obs.values

    def _take_action(self, action, not_execute = False):
        # Set the current price to a random price within the time step
        # dim(self.actual_price) = [n,6], dim(action) = [1, n+1]

        # self.actual_price[pd.isna(self.actual_price)] = self.prev_buyNhold_price[pd.isna(self.actual_price)]
        rebalanced_weight = self.action_reference[action]
        inventory_value = self.actual_price * self.inventory_number
        rebalanced_inv_value = rebalanced_weight*np.sum(inventory_value) # Calculated rebalanced value, did not consider comission
        
        '''
        Updated on 20 Feb.
        1. Calculate the total cash from sell (including "selling cash", which means put the cash into the pool)
        2. Calculate the amount of cash used to purchase asset (including "buying cash")
        3. Calculate the number for buying
        4. Update the inventory
        '''

        # Sell first, then use the cash to buy
        sell_value = rebalanced_inv_value - inventory_value
        sell_value[sell_value > 0] = 0
        sell_value *= -1
        
        sell_number = sell_value / self.actual_price
        inv_number_after_sell = self.inventory_number - sell_number
        cash_from_sale = np.sum(sell_value) * (1-COMMISSION_FEE)

        # Use the cash from sale to buy the stocks
        buy_value = rebalanced_inv_value - inventory_value
        buy_value[buy_value < 0] = 0
        buy_weight = buy_value / np.sum(buy_value) # Normalize by the sum -> Indicate the percentage of money it can use
        buy_number = cash_from_sale*(1-COMMISSION_FEE) * buy_weight / self.actual_price
        self.proposed_inv_num = inv_number_after_sell + buy_number
        
        if not not_execute:
            self._update_params(sell_value, buy_value)

    def _update_params(self, sell_value, buy_value, cashout=False):
        '''
        Updates self.total_sales_value; self.inventory_number; self.prev_inventory_num;
                self.cost_basis; self.prev_net_worth; self.net_worth;
                self.prev_total_net_worth; self.total_net_worth; self.max_net_worth;                        
        '''
        if not cashout:
            self.total_sales_value += sell_value
            prev_cost = self.cost_basis * self.inventory_number
            self.inventory_number = self.proposed_inv_num

            if np.isnan(self.inventory_number).any():
                self.inventory_number = self.prev_inventory_num
            else:
                self.prev_inventory_num = self.inventory_number

            a = (prev_cost - sell_value + buy_value)
            b = self.inventory_number
            self.cost_basis = np.divide(a, b, out=np.zeros_like(a), where=b!=0)

            self.cost_basis[pd.isna(self.cost_basis)] = 0
            self.cost_basis[np.isinf(self.cost_basis)] = 0
        
        else:
            self.prev_inventory_num = self.inventory_number
            self.inventory_number = np.array([0.0]*self.market_number)
            self.cost_basis = np.array([0.0]*self.market_number)
            self.total_sales_value += self.prev_inventory_num*self.actual_price*(1-COMMISSION_FEE)

        self.prev_net_worth = self.net_worth
        self.net_worth = self.inventory_number * self.actual_price

        self.prev_total_net_worth = self.total_net_worth
        self.total_net_worth = np.sum(self.net_worth) + self.cash
        if self.total_net_worth > self.max_net_worth:
            self.max_net_worth = self.total_net_worth


        # if np.isnan(self.inventory_number).any():
        #     raise Exception("Inv NAN WARNING!")

    def step(self, one_hot_action):
        '''
        Taking the action in the environment.
        Action: [0,0,0,0]-[1,1,1,1] -> a decision in 0-3
        Action 0: 80% High, 10% Mid, 10% Low;
        Action 1: 10% High, 80% Mid, 10% Low;
        Action 2: 45% High, 45% Mid, 10% Low;
        Action 3: 10% High, 10% Mid, 80% Low.
        '''
        # 0. Determine whether can take action: Allow to take action for at freq of 1 month
        self.open_for_transaction = self.current_step >= (self.prev_action_step + self.action_freq)
        
        # 1. Translate one-hot action into int.
        if np.isnan(one_hot_action).any():
            one_hot_action = np.array([0.0, 0.0, 0.0, 0.0])
        
        self.prev_action = self.current_action
        self.current_action = one_hot_action
        action = np.argmax(one_hot_action) 
        # Returns the index of the max number, all same return 0 
        # => Choose Action 0-3, Default 0
        
        self.actual_price = np.array([df.loc[self.current_step, "Actual Price"] for df in self.df_list], dtype=np.float64)     

        # 2. Take Action. 

        if ((np.sum(one_hot_action) > 0) and 
            (self.open_for_transaction or (self.current_step >= (self.prev_action_step + self.action_freq*2)))):      # In case not reacting for too long
            
            self._take_action(action, not_execute=False)
            self.prev_action_step = self.current_step
            self.open_for_transaction = False
            '''
                Updates self.balance, self.cost_basis, self.shares_held,
                        self.total_shares_sold, self.total_sales_value,
                        self.net_worth, self.max_net_worth, 
            '''

        else:
            self._take_action(action, not_execute=True)
        
        # 3. Get the close price for TODAY and calculate profit        
        self._update_buyNhold()

        profit = self.total_net_worth - INITIAL_ACCOUNT_BALANCE
        actual_profit = self.total_net_worth - self.buyNhold_balance
    

        # ============== Calculate Rewards ==============
        
        '''
        1. See the total value in the future for this action
        2. See the total value in the future if we dont take action
        '''
        # Calculate Benchmark Performances
        # 1. Get the future price
        future_step = self.current_step+self.reward_wait
        future_step = min(future_step, self.end_step-1)
        # future_price = [df.loc[future_step, "Price"] for df in self.df_list]
        future_price = []

        for df in self.df_list:
            try:
                price = df.loc[future_step, "Actual Price"]
                future_price.append(price)
            except Exception as e:
                print(e)
                future_price.append(df['Actual Price'].values[-1])
                
        
        future_price = np.array(future_price, dtype=np.float64)

        passive_FV = self.prev_inventory_num * future_price # FV: Future Value
        current_FV = self.proposed_inv_num * future_price

        delay_modifier = 1-(self.current_step / len(self.intersect_dates)*0.5)
        delta_FV = np.sum(current_FV - passive_FV)/np.sum(passive_FV)
        if np.isnan(delta_FV) or np.isinf(delta_FV):
            delta_FV = 0
        change_reward = delta_FV*delay_modifier
        profit_reward = self.total_net_worth / 10000000
        
        reward = change_reward + profit_reward
        if reward==np.inf:
            print("INF Reward")
        
        # 3. Update Next Date: If reaches the end then go back to time 0.
        self._update_current_step()
            # IMPORTANT: From now on, the current_step becomes TOMORROW To Keep the current_step undiscovered
        
        # 4. All the CRISIS AVOIDANCE STUFF
        if self.crisis_detection:             
            # Check whether the next current_step any crisis happens
            skip = True # skip: whether all three has a max_drawback 
            while skip == True:
                # Check the maximum drawdown
                if self.finished:
                    break
                drawdown_count = 0
                for df in self.df_list: # Check if the max monthly drawdown >= 20%
                    tmp_df = df['daily_Drawdown'].loc[np.min(self.current_step-self.MDD_window,0):self.current_step]
                    if np.min(tmp_df) <= -1*self.drawdown_threshold: # negative number
                        drawdown_count += 1
                
                if drawdown_count == self.market_number: # Crisis happens in today: All the markets go off
                    if not self.cash_out_trigger: # Happens for the first time: 
                        # !!!!!!!!!!!! CASH OUT !!!!!!!!!!!!
                        print("CASH OUT", self.df_list[0]['Date'][self.current_step])
                        self.actual_price = np.array([random.uniform(df.loc[self.current_step, "Low"],
                                                        df.loc[self.current_step, "High"]) for df in self.df_list], dtype=np.float64)     
                        self.cash = np.sum(self.inventory_number*self.actual_price*(1-COMMISSION_FEE))
                        self._update_params(sell_value = None, buy_value = None, cashout=True)
                        self.cash_out_trigger = True

                    self._update_buyNhold()
                    self._update_current_step()
                
                else: # If today no crisis:
                    if self.cash_out_trigger: # If previously have crisis: FINALLY ENDS
                        # !!!!!!!!!!!! BUY IN !!!!!!!!!!!!
                        print("BUY IN", self.df_list[0]['Date'][self.current_step])
                        self.actual_price = np.array([df.loc[self.current_step, "Actual Price"] for df in self.df_list], dtype=np.float64)     
                        buy_value = [self.cash*(1-COMMISSION_FEE)/self.market_number]*self.market_number
                        self.proposed_inv_num = buy_value/self.actual_price
                        self.cash = 0
                        self._update_params(sell_value = np.array([0.0]*self.market_number), buy_value = buy_value)
                        self._update_buyNhold()
                        self._update_current_step()

                    skip = False
                    self.cash_out_trigger = False
        
        # OpenAI will reset if done==True
        done = (self.total_net_worth <= 0) or self.finished
        obs = self._next_observation()        
        info = {"profit": profit, "total_shares_sold": self.total_sales_value, "actual_profit": actual_profit}
        return (obs, reward, done, info)

    def _update_buyNhold(self):
        close_prices = [df.loc[self.current_step, "Actual Price"] for df in self.df_list]
        close_prices = np.array(close_prices, dtype=np.float64)

        self.prev_buyNhold_balance = self.buyNhold_balance
        self.buyNhold_balance = np.sum(
            self.init_buyNhold_number * close_prices)
        self.prev_buyNhold_price = close_prices
        

    def _update_current_step(self):
        last_step = self.end_step-self.reward_wait-self.action_freq-1
        if self.current_step >= last_step:
            if self.training:
                self.current_step = self.window_size  # Going back to time 0

                close_prices = [df.loc[self.current_step, "Price"] for df in self.df_list]
                close_prices = np.array(close_prices, dtype=np.float64)
                self.inventory_number = self.net_worth/close_prices
                self.prev_inventory_num = self.inventory_number
                self.cost_basis = close_prices
                self.open_for_transaction = True

            else:  # if is testing: Stop iteratioin
                self.current_step = last_step
                self.finished = True
        else:
            # 1. Execute TODAY's Action
            self.current_step += 1


    def render(self, mode='human', close=False, afterStep=True):
        '''
        afterStep: if is rendered after the step function, the current_step should -=1.
        '''
        if afterStep:
            todayDate = self.intersect_dates[self.current_step-1]
        else:
            todayDate = self.intersect_dates[self.current_step]
        if mode == 'human':
            # Render the environment to the screen
            profit = self.total_net_worth - INITIAL_ACCOUNT_BALANCE

            print(f'Date: {todayDate}')
            # print(f'Balance: {self.cash}')
            print(
                f'Shares held: {self.inventory_number}')
            print(
                f'Avg cost for held shares: {self.cost_basis} (Total sales value: {self.total_sales_value})')
            print(
                f'Net worth: {self.net_worth} (Total net worth: {self.total_net_worth})')
            print(f'Profit: {profit}')

        elif mode == 'detail':  # Want to add all transaction details
            net_worth_delta = self.total_net_worth - self.prev_total_net_worth
            buyNhold_delta = self.buyNhold_balance - self.prev_buyNhold_balance
            return {
                "date": todayDate,
                "actual_price": self.actual_price,
                "action": self.current_action,
                "inventory": self.net_worth,
                "shares_held": self.inventory_number,
                "net_worth": self.total_net_worth,
                "net_worth_delta": net_worth_delta,
                "buyNhold_balance": self.buyNhold_balance,
                "buyNhold_delta": buyNhold_delta,
                "actual_profit": self.total_net_worth - self.buyNhold_balance,
                "progress": (net_worth_delta+1)/(buyNhold_delta+1)
            }
