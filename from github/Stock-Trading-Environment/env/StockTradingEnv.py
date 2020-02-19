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
COMMISSION_FEE = 0.008

INITIAL_ACCOUNT_BALANCE = 10000


class StockTradingEnv(gym.Env):
    # metadata = {'render.modes': ['human']}

    def _getClosestDateAfter(df, date):
        s = pd.to_datetime(df.index.to_series()) - pd.to_datetime(startDate)
        return s[s >= pd.Timedelta(0)].idxmin()

    def __init__(self, df_list, startDate, isTraining=True):
        super(StockTradingEnv, self).__init__()

        self.training = isTraining
        self.window_size = 6
        self.df_list = []
        for df in df_list:
            df.reset_index(drop=True)
            df.set_index('Date', inplace=True) # For Multiple Markets: Use Date as index
            self.df_list.append(df)

        self.start_date = min([self._getClosestDateAfter(df, startDate) for df in self.df_list])
        self.current_date = self.start_date + pd.Timedelta(days = self.window_size) # For Multiple Markets: Replace current_step with current_date

        market_number = len(df_list)+1  # For Multiple Markets: Adding the CASH to the action
        lower_bond = [0]*market_number
        upper_bond = [1]*market_number
        self.action_space = spaces.Box(
            low=np.array(lower_bond), high=np.array(upper_bond), dtype=np.float16)
        # Give weight to each and we take the average later, the last Asset is the CASH

        # Prices contains the OHCL values for the last six prices
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(6, 6), dtype=np.float16)

    def _next_observation(self):
        '''
        The _next_observation method compiles the stock data for the last five time steps, 
        appends the agent’s account information, and scales all the values to between 0 and 1.
        '''

        # self.current_step is defined in reset method,
        # We assume the current_step is TODAY (BEFORE FINAL), which means we only know infomation till YESTERDAY ENDS.
        obs_list = []
        
        for i, df in enumerate(self.df_list):
            frame = np.array([
                df.loc[pd.date_range(self.current_date-pd.Timedelta(days=self.window_size), 
                    self.current_date-pd.Timedelta(days=1)),
                    'Open'].values / MAX_SHARE_PRICE,
                
                df.loc[pd.date_range(self.current_date-pd.Timedelta(days=self.window_size), 
                    self.current_date-pd.Timedelta(days=1)),
                    'High'].values / MAX_SHARE_PRICE,
                
                df.loc[pd.date_range(self.current_date-pd.Timedelta(days=self.window_size), 
                    self.current_date-pd.Timedelta(days=1)),
                    'Low'].values / MAX_SHARE_PRICE,

                df.loc[pd.date_range(self.current_date-pd.Timedelta(days=self.window_size), 
                    self.current_date-pd.Timedelta(days=1)),
                    'Price'].values / MAX_SHARE_PRICE,

                df.loc[pd.date_range(self.current_date-pd.Timedelta(days=self.window_size), 
                    self.current_date-pd.Timedelta(days=1)),
                    'Vol'].values / MAX_NUM_SHARES,
            ])

            # Append additional data and scale each value to between 0-1
            obs = np.append(frame, [[
                self.cash / MAX_ACCOUNT_BALANCE,
                self.total_net_worth / MAX_ACCOUNT_BALANCE,
                self.net_worth[i] / MAX_ACCOUNT_BALANCE,
                self.shares_held[i] / MAX_NUM_SHARES,
                self.cost_basis[i] / MAX_SHARE_PRICE,
                self.total_shares_sold[i] / MAX_NUM_SHARES,
            ]], axis=0)

            obs_list.append(obs)
        

        cash_obs = np.array([
                [1]*self.window_size / MAX_SHARE_PRICE,
                [1]*self.window_size / MAX_SHARE_PRICE,
                [1]*self.window_size / MAX_SHARE_PRICE,
                [1]*self.window_size / MAX_SHARE_PRICE,
                [1]*self.window_size
            ])
            
        cash_obs = np.append(cash_obs, [[
                self.cash / MAX_ACCOUNT_BALANCE,
                self.total_net_worth / MAX_ACCOUNT_BALANCE,
                self.cash / MAX_ACCOUNT_BALANCE,
                self.cash / MAX_NUM_SHARES,
                1 / MAX_SHARE_PRICE,
                self.cash / MAX_NUM_SHARES,
            ]], axis=0)

        obs_list.append(cash_obs)

        return np.array(obs_list)

    def _take_action(self, action):
        # Set the current price to a random price within the time step
        
        self.actual_price = np.array([
            random.uniform(df.loc[self.current_date, "Low"], df.loc[self.current_date, "High"]) 
            if self.current_date in df.index
            else np.nan 
            for df in self.df_list]) # e.g.[np.nan, ]

        tradable_stock = np.isnan(self.actual_price)
        available_shares_value = self.actual_price * self.shares_held # dim:n, those with no price will be nan
        available_total_value = np.nansum(available_shares_value)
        available_action = action * (tradable_stock * -1 + 1)
        available_action_weighted = available_action/np.nansum(available_action)


        # dim(self.actual_price) = [n,6], dim(action) = [1, n+1]




    '''
        action_type = action[0]
        buyAmount = action[1]
        sellAmount = action[2]

        if action_type < 1:
            # [0,1): Buy amount % of balance in shares
            cash_spend = self.balance * buyAmount
            if cash_spend < 0.01*self.net_worth:  # Not executing this transaction
                buyAmount = 0
                cash_spend = 0
                self.current_action = 0
            else:
                shares_bought = cash_spend / \
                    (self.actual_price*(1+COMMISSION_FEE))
                self.current_action = shares_bought
                prev_cost = self.cost_basis * self.shares_held

                self.balance -= cash_spend
                self.cost_basis = (
                    prev_cost + cash_spend) / (self.shares_held + shares_bought)
                self.shares_held += shares_bought

        elif action_type < 2:
            # [1,2): Sell amount % of shares held

            shares_sold = self.shares_held * sellAmount
            cash_get = shares_sold*self.actual_price*(1-COMMISSION_FEE)
            if cash_get < 0.001*self.net_worth:  # Not executing this transaction
                sellAmount = 0
                shares_sold = 0
                cash_get = 0
                self.current_action = 0
            else:
                self.current_action = shares_sold*-1
                self.balance += shares_sold * self.actual_price
                self.shares_held -= shares_sold
                self.total_shares_sold += shares_sold
                self.total_sales_value += shares_sold * self.actual_price
        else:  # [2,3): Hold
            self.current_action = 0

        self.prev_net_worth = self.net_worth
        self.net_worth = self.balance + self.shares_held * self.actual_price

        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth

        if self.shares_held == 0:
            self.cost_basis = 0
    '''

    def step(self, action):
        '''
        Next, our environment needs to be able to take a step. 
        At each step we will take the specified action (chosen by our model), 
        calculate the reward, and return the next observation.
        '''
        # 1. Determine TODAY's Date (For training)
        if self.current_step > len(self.df.loc[:, 'Open'].values) - 2:
            # if self.training:
            if self.training:
                self._take_action(action)
                self.current_step = self.window_size  # Going back to time 0
            else:  # if is testing
                if not self.finished:
                    self.finished = True
                    print("$$$$$$$$$$$ CASH OUT at time " +
                          str(self.df.loc[self.current_step, "Date"]) + "$$$$$$$$$$$")
                    # SELL EVERYTHING on the last day
                    action = np.array([1, 0, 1])
                    self._take_action(action)
                    self.current_step += 1
                else:
                    self.finished_twice = True
        else:
            # 1. Execute TODAY's Action
            self._take_action(action)
            '''
            Updates self.balance, self.cost_basis, self.shares_held,
                    self.total_shares_sold, self.total_sales_value,
                    self.net_worth, self.max_net_worth, 
            '''
            self.current_step += 1
            # ****IMPORTANT: From now on, the current_step becomes TOMORROW****
            # Keep the current_step undiscovered

        '''
        We want to incentivize profit that is sustained over long periods of time. 
        At each step, we will set the reward to the account balance multiplied by 
        some fraction of the number of time steps so far.

        The purpose of this is to delay rewarding the agent too fast in the early stages 
        and allow it to explore sufficiently before optimizing a single strategy too deeply. 
        It will also reward agents that maintain a higher balance for longer, 
        rather than those who rapidly gain money using unsustainable strategies.
        '''
        self.prev_buyNhold_balance = self.buyNhold_balance
        self.buyNhold_balance = self.init_buyNhold_amount * \
            self.df.loc[self.current_step-1, "Price"]

        profit = self.total_net_worth - INITIAL_ACCOUNT_BALANCE
        actual_profit = self.total_net_worth - self.buyNhold_balance

        delay_modifier = (self.current_step / MAX_STEPS)
        # reward = self.balance * delay_modifier  # Original Version
        # reward = actual_profit * delay_modifier  # Use Actual Net Profit

        net_worth_delta = self.total_net_worth - self.prev_net_worth
        buyNhold_delta = self.buyNhold_balance - self.prev_buyNhold_balance
        reward = (net_worth_delta+1)/(buyNhold_delta+1)

        # OpenAI will reset if done==True
        done = (self.total_net_worth <= 0) or self.finished_twice
        if not self.finished:
            obs = self._next_observation()
        else:
            self.current_step -= 1
            obs = self._next_observation()
            self.current_step += 1

        if not self.finished_twice:
            info = {"profit": profit, "total_shares_sold": self.total_shares_sold,
                    "actual_profit": actual_profit}
        else:
            info = {"profit": 0, "total_shares_sold": 0, "actual_profit": 0}
        return (obs, reward, done, info)

    def reset(self):
        # Reset the state of the environment to an initial state
        self.cash = INITIAL_ACCOUNT_BALANCE
        self.total_net_worth = INITIAL_ACCOUNT_BALANCE
        self.max_net_worth = INITIAL_ACCOUNT_BALANCE
        self.shares_held = 0
        self.cost_basis = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0
        self.current_action = 0
        self.prev_net_worth = INITIAL_ACCOUNT_BALANCE
        self.prev_buyNhold_balance = 0
        self.finished = False
        self.finished_twice = False

        # Set the current step to a random point within the data frame
        # We set the current step to a random point within the data frame, because it essentially gives our agent’s more unique experiences from the same data set.
        if self.training:
            self.current_step = random.randint(
                self.window_size, len(self.df.loc[:, 'Open'].values))
        else:
            self.current_step = self.window_size

        self.init_buyNhold_amount = INITIAL_ACCOUNT_BALANCE / \
            self.df.loc[self.current_step, "Price"]
        self.buyNhold_balance = INITIAL_ACCOUNT_BALANCE

        return self._next_observation()

    def render(self, mode='human', close=False, afterStep=True):
        '''
        afterStep: if is rendered after the step function, the current_step should -=1.
        '''
        todayStep = self.current_step
        if afterStep:
            todayStep -= 1

        if mode == 'human':
            # Render the environment to the screen
            profit = self.total_net_worth - INITIAL_ACCOUNT_BALANCE

            print(f'Step: {todayStep}')
            print(f'Balance: {self.balance}')
            print(
                f'Shares held: {self.shares_held} (Total sold: {self.total_shares_sold})')
            print(
                f'Avg cost for held shares: {self.cost_basis} (Total sales value: {self.total_sales_value})')
            print(
                f'Net worth: {self.net_worth} (Max net worth: {self.max_net_worth})')
            print(f'Profit: {profit}')

        elif mode == 'detail':  # Want to add all transaction details
            net_worth_delta = self.total_net_worth - self.prev_net_worth
            buyNhold_delta = self.buyNhold_balance - self.prev_buyNhold_balance
            return {
                "step": todayStep,
                "date": self.df.loc[todayStep, "Date"],
                "actual_price": self.actual_price,
                "action": self.current_action,
                "shares_held": self.shares_held,
                "net_worth": self.total_net_worth,
                "net_worth_delta": net_worth_delta,
                "buyNhold_balance": self.buyNhold_balance,
                "buyNhold_delta": buyNhold_delta,
                "actual_profit": self.total_net_worth - self.buyNhold_balance,
                "progress": (net_worth_delta+1)/(buyNhold_delta+1)
            }
