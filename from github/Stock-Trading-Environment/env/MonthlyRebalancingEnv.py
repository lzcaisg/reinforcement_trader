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


class RebalancingEnv(gym.Env):
    # metadata = {'render.modes': ['human']}

    def __init__(self, df_dict, col_list, isTraining=True):
        super(RebalancingEnv, self).__init__()

        self.training = isTraining
        self.col_list = col_list
        self.window_size = 6
        self.wait_days = 10 
        self.punish_no_action = False
        # wait_days: Number of days need to wait for determine the reward.
        self.action_freq = 7
        # Take action for every 7 days.
        self.df_list = []
        # Determine the number of markets
        self.market_number = len(df_dict)

        # 1. Get the intersect dates from different stocks
        tmp_df = df_dict["high"].dropna()
        self.intersect_dates = tmp_df['Date']
        for key in df_dict:
            df = df_dict[key].dropna()
            self.intersect_dates = np.intersect1d(self.intersect_dates, df['Date'])
        # Remove all NAN in the df
        
        self.start_date = np.min(self.intersect_dates)
        self.end_date = np.max(self.intersect_dates)
        self.prev_action_step = 0

        # 2. Add only the common dates
        for key in ['high', 'mid', 'low']:
            df = df_dict[key].dropna()
            self.df_list.append(df[df['Date'].isin(self.intersect_dates)].reset_index(drop=True))

        # 3. Action Space: [0,0,0,0]->[1,1,1,1]: Four Actions
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0]), high=np.array([1.0, 1.0, 1.0, 1.0]), dtype=np.float16)

        # 4. Observation Space
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(6, 6), dtype=np.float16)

    def _next_observation(self):
        '''
        The _next_observation method provices observations in the format of:
        States = (EMA_h, MACD_h, EMA_m, MACD_m, delta_time)
        Refer to the report for detailed explanation.
        '''
        # self.current_step is defined in reset method,
        # We assume the current_step is TODAY (BEFORE FINAL), which means we only know infomation till YESTERDAY ENDS.
        # today_date = self.intersect_dates(self.current_step)
        high_obs = self.df_list[0][self.col_list][self.current_step-self.window_size:self.current_step]
        mid_obs  = self.df_list[1][self.col_list][self.current_step-self.window_size:self.current_step]

        obs = pd.concat([high_obs, mid_obs], axis=1, sort=False)
        obs.columns = ['EMA_h', 'MACD_diff_h', 'delta_time_h', 'EMA_m', 'MACD_diff_m', 'delta_time_m']
        obs.reset_index(inplace=True, drop=True)
        obs[['EMA_h', 'EMA_m']] /= obs[['EMA_h', 'EMA_m']].iloc[0]
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
            
            self.prev_net_worth = self.net_worth
            self.net_worth = self.inventory_number * self.actual_price
            
            self.prev_total_net_worth = self.total_net_worth
            self.total_net_worth = np.sum(self.net_worth)
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
        self.open_for_transaction = self.current_step > self.prev_action_step + 20
        
        # 1. Translate one-hot action into int.
        if np.isnan(one_hot_action).any():
            one_hot_action = np.array([0.0, 0.0, 0.0, 0.0])
        
        self.prev_action = self.current_action
        self.current_action = one_hot_action
        action = np.argmax(one_hot_action) 
        # Returns the index of the max number, all same return 0 
        # => Choose Action 0-3, Default 0
        
        self.actual_price = np.array([random.uniform(df.loc[self.current_step, "Low"],
                                                     df.loc[self.current_step, "High"]) for df in self.df_list], dtype=np.float64)     

        # 2. Take Action. 
        if (np.sum(one_hot_action) > 0) and self.open_for_transaction:
            self._take_action(action, not_execute=False)
            self.prev_action_step = self.current_step
            self.open_for_transaction = False
            '''
                Updates self.balance, self.cost_basis, self.shares_held,
                        self.total_shares_sold, self.total_sales_value,
                        self.net_worth, self.max_net_worth, 
            '''

            # 3. Get the close price for TODAY and calculate profit        
            close_prices = [df.loc[self.current_step, "Price"] for df in self.df_list]
            close_prices = np.array(close_prices, dtype=np.float64)

            self.prev_buyNhold_balance = self.buyNhold_balance
            self.buyNhold_balance = np.sum(
                self.init_buyNhold_number * close_prices)
            self.prev_buyNhold_price = close_prices

        else:
            self._take_action(action, not_execute=True)

        profit = self.total_net_worth - INITIAL_ACCOUNT_BALANCE
        actual_profit = self.total_net_worth - self.buyNhold_balance
    

        # ============== Calculate Rewards ==============
        
        '''
        1. See the total value in the future for this action
        2. See the total value in the future if we dont take action
        '''
        # Calculate Benchmark Performances
        # 1. Get the future price
        future_step = self.current_step+self.wait_days
        max_lenth = min([len(df['Price']) for df in self.df_list])
        future_step = min(future_step, max_lenth)
        future_price = [df.loc[self.current_step+self.wait_days, "Price"] for df in self.df_list]
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
            print("INF")
        
        if self.punish_no_action:
            if np.sum(one_hot_action) != 0:
                self.zero_action_count = 0
            else:
                self.zero_action_count += 1
                if self.zero_action_count >= 3:
                    reward = 0


        # 3. Update Next Date: If reaches the end then go back to time 0.
        last_step = len(self.intersect_dates)-self.wait_days-self.action_freq-1
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
            
            # ****IMPORTANT: From now on, the current_step becomes TOMORROW****
            # Keep the current_step undiscovered
        
        
        # OpenAI will reset if done==True
        done = (self.total_net_worth <= 0) or self.finished
        obs = self._next_observation()
        
        # if not self.finished:
        #     obs = self._next_observation()
        # else: # If already finished: 
        #     self.current_step -= 1
            # obs = self._next_observation()
            # self.current_step += 1

        
        info = {"profit": profit, "total_shares_sold": self.total_sales_value, "actual_profit": actual_profit}
        return (obs, reward, done, info)

    def reset(self):
        # Reset the state of the environment to an initial state
        
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

        self.total_net_worth = INITIAL_ACCOUNT_BALANCE
        self.prev_total_net_worth = INITIAL_ACCOUNT_BALANCE
        self.max_net_worth = INITIAL_ACCOUNT_BALANCE
        self.current_step = 0
        self.zero_action_count = 0

        self.prev_buyNhold_balance = 0
        self.finished = False
        self.finished_twice = False

        self.net_worth = np.array(
            [INITIAL_ACCOUNT_BALANCE / self.market_number] * self.market_number, 
            dtype=np.float64)

        
        
        self.total_sales_value = np.array([0.0] * self.market_number)
        self.prev_net_worth = self.net_worth
        
        self.prev_action = np.array([0.0,0.0,0.0,0.0])
        self.current_action = self.prev_action

        self.prev_action_step = 0 

        # Set the current step to a random point within the data frame
        # We set the current step to a random point within the data frame, because it essentially gives our agent’s more unique experiences from the same data set.
        if self.training:
            days_range = len(self.intersect_dates)
            rand_days = random.randint(self.window_size, days_range - 1)
            self.current_step = rand_days
        else:
            self.current_step = self.window_size

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
