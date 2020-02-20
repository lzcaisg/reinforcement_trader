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
    

    def __init__(self, df_list, isTraining=True):
        super(StockTradingEnv, self).__init__()

        self.training = isTraining
        self.window_size = 6
        self.df_list = []
        for df in df_list:
            df.reset_index(drop=True)
            df.set_index('Date', inplace=True) # For Multiple Markets: Use Date as index
            self.df_list.append(df)

        self.intersect_date = df_list[0].index
        for df in df_list:
            self.intersect_date = np.intersect1d(self.intersect_date, df.index)
        
        self.start_date = np.min(self.intersect_date)
        self.end_date = np.max(self.intersect_date)
        
        market_number = len(df_list)+1  # For Multiple Markets: Adding the CASH to the action
        lower_bond = [0.0]*market_number
        upper_bond = [1.0]*market_number
        self.action_space = spaces.Box(
            low=np.array(lower_bond), high=np.array(upper_bond), dtype=np.float16)
        # Give weight to each and we take the average later, the last Asset is the CASH

        # Prices contains the OHCL values for the last six prices
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(market_number, 6, 6), dtype=np.float16)

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
            ], dtype=np.float64)

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
                np.array([1]*self.window_size) / MAX_SHARE_PRICE,
                np.array([1]*self.window_size) / MAX_SHARE_PRICE,
                np.array([1]*self.window_size) / MAX_SHARE_PRICE,
                np.array([1]*self.window_size) / MAX_SHARE_PRICE,
                np.array([1]*self.window_size)
            ], dtype=np.float64)

        cash_obs = np.stack(cash_obs)

        cash_obs = np.append(cash_obs, [[
                self.cash / MAX_ACCOUNT_BALANCE,
                self.total_net_worth / MAX_ACCOUNT_BALANCE,
                self.cash / MAX_ACCOUNT_BALANCE,
                self.cash / MAX_NUM_SHARES,
                1 / MAX_SHARE_PRICE,
                self.cash / MAX_NUM_SHARES,
            ]], axis=0)

        obs_list.append(cash_obs)

        return np.array(obs_list, dtype=np.float64)

    def _take_action(self, action):
        # Set the current price to a random price within the time step
        # dim(self.actual_price) = [n,6], dim(action) = [1, n+1]
        
        self.actual_price = np.array([random.uniform(df.loc[self.current_date, "Low"], df.loc[self.current_date, "High"]) if self.current_date in df.index else np.nan for df in self.df_list], dtype=np.float64) # e.g.[np.nan, ]
        self.actual_price = np.append(self.actual_price, 1)
        # Add CASH price = 1, now dim=n+1, MAY HAVE NAN

        untradable_stock = pd.isnull(self.actual_price)
        available_shares_value = self.actual_price * self.shares_held # dim:n+1, those with no price will be nan
        available_action = action * (untradable_stock * -1 + 1) # For actions being nan, we should not change it

        available_action_weighted = available_action/np.nansum(available_action)
        available_shares_value_weighted = available_shares_value/np.nansum(available_shares_value)
        delta_weight = available_action_weighted - available_shares_value_weighted
        delta_weight[pd.isnull(delta_weight)] = 0

        delta_value = delta_weight * np.nansum(available_shares_value) # Buy or sell stocks worth of $x, exclusive of COMMISSION FEE
        delta_cash = delta_value[-1]
        delta_value[-1] = 0 # Remove the cash delta to avoid counting commission fee twice
        # DELTA VALUE IS NOT FINAL!! Net of COMMISSION FEE!!

        sell_value = delta_value * (delta_value < 0)
        buy_value = delta_value * (delta_value > 0)
        
        sell_value *= -1
        sell_number = sell_value / (self.actual_price) # delta Cash == value, LESS CASH IS RECEIVED
        cash_received_from_sale = np.nansum(sell_value*(1-COMMISSION_FEE))
        
        buy_number = buy_value / (self.actual_price * (1+COMMISSION_FEE)) # delta Cash == value, LESS STOCK IS BOUGHT due to the COMMISSION FEE
        cash_paid_for_buying = np.nansum(buy_value)
        buy_value = buy_number * self.actual_price

        delta_cash = cash_received_from_sale - cash_paid_for_buying
        delta_value = buy_value-sell_value
        delta_value[pd.isnull(delta_value)] = 0

        delta_number = buy_number + sell_number
        delta_number[-1] = delta_cash
        delta_number[pd.isnull(delta_number)] = 0

        prev_cost = self.cost_basis * self.shares_held
        self.cost_basis = (prev_cost + delta_value) / (self.shares_held + delta_number)
        
        self.shares_held += delta_number
        self.cash = self.shares_held[-1]
        self.total_shares_sold += sell_number
        self.prev_net_worth = self.net_worth
        self.net_worth = self.shares_held * self.actual_price
        self.net_worth[pd.isnull(self.net_worth)] = self.prev_net_worth[pd.isnull(self.net_worth)]
        # If the asset is not for trading, use the previous value
        self.prev_total_net_worth = self.total_net_worth
        self.total_net_worth = np.sum(self.net_worth)
        if self.total_net_worth > self.max_net_worth:
            self.max_net_worth = self.total_net_worth



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
        if self.current_date >= self.end_date:
            # if self.training:
            if self.training:
                self._take_action(action)
                self.current_date = self.start_date  # Going back to time 0
            else:  # if is testing
                if not self.finished:
                    self.finished = True
                    print("$$$$$$$$$$$ CASH OUT at time " +
                          str(self.current_date) + "$$$$$$$$$$$")
                    # SELL EVERYTHING on the last day
                    action = np.array(([0]*len(self.df_list)).append(1)) #[0,0,0,...,1]: Cash Out
                    self._take_action(action)
                    self.current_date += pd.Timedelta(days = 1)
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
            self.current_date += pd.Timedelta(days = 1)
            # ****IMPORTANT: From now on, the current_date becomes TOMORROW****
            # Keep the current_date undiscovered

        '''
        We want to incentivize profit that is sustained over long periods of time. 
        At each step, we will set the reward to the account balance multiplied by 
        some fraction of the number of time steps so far.

        The purpose of this is to delay rewarding the agent too fast in the early stages 
        and allow it to explore sufficiently before optimizing a single strategy too deeply. 
        It will also reward agents that maintain a higher balance for longer, 
        rather than those who rapidly gain money using unsustainable strategies.
        '''
        
        close_prices = []
        for df in self.df_list:
            if (self.current_date-pd.Timedelta(days = 1)) in df.index:
                close_prices.append(df.loc[self.current_date-pd.Timedelta(days = 1), "Price"])
            else:
                close_prices.append(np.nan)
        close_prices.append(1)
        close_prices = np.array(close_prices, dtype=np.float64)
        close_prices[pd.isnull(close_prices)] = self.prev_buyNhold_price[pd.isnull(close_prices)]
        
        self.prev_buyNhold_balance = self.buyNhold_balance
        self.buyNhold_balance = np.sum(self.init_buyNhold_amount * close_prices)
        self.prev_buyNhold_price = close_prices

        profit = self.total_net_worth - INITIAL_ACCOUNT_BALANCE
        actual_profit = self.total_net_worth - self.buyNhold_balance

        # delay_modifier = (self.current_step / MAX_STEPS)
        # reward = self.balance * delay_modifier  # Original Version
        # reward = actual_profit * delay_modifier  # Use Actual Net Profit

        total_net_worth_delta = self.total_net_worth - self.prev_total_net_worth
        buyNhold_delta = self.buyNhold_balance - self.prev_buyNhold_balance
        
        reward = (total_net_worth_delta+1)/(buyNhold_delta+1) # TODO: NEED TO Reengineer!!!

        # OpenAI will reset if done==True
        done = (self.total_net_worth <= 0) or self.finished_twice
        
        
        if not self.finished:
            obs = self._next_observation()
        else:
            self.current_date -= pd.Timedelta(days = 1)
            obs = self._next_observation()
            self.current_date += pd.Timedelta(days = 1)

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
        self.prev_total_net_worth = INITIAL_ACCOUNT_BALANCE
        self.max_net_worth = INITIAL_ACCOUNT_BALANCE
        
        self.prev_buyNhold_balance = 0
        self.finished = False
        self.finished_twice = False
        
        self.net_worth = [0]*len(self.df_list)
        self.net_worth.append(INITIAL_ACCOUNT_BALANCE)
        self.net_worth = np.array(self.net_worth, dtype=np.float64)
        
        self.shares_held = self.net_worth
        self.total_shares_sold = self.net_worth
        self.prev_net_worth = self.net_worth
        
        self.cost_basis = [0]*len(self.df_list)
        self.cost_basis.append(1)
        self.cost_basis = np.array(self.cost_basis, dtype=np.float64)


        self.total_sales_value = np.array([0]*(len(self.df_list)+1), dtype=np.float64)
        self.current_action = np.array([0]*(len(self.df_list)+1), dtype=np.float64)
        


        # Set the current step to a random point within the data frame
        # We set the current step to a random point within the data frame, because it essentially gives our agent’s more unique experiences from the same data set.
        if self.training:
            days_range = int((self.end_date - self.start_date).astype('timedelta64[D]') / np.timedelta64(1, 'D'))
            rand_days = random.randint(self.window_size, days_range - 1)
            self.current_date = self.start_date + pd.Timedelta(days = rand_days) 
        else:
            self.current_date = self.start_date + pd.Timedelta(days = self.window_size) # For Multiple Markets: Replace current_step with current_date

        init_price = [df.loc[self.start_date, "Price"] for df in self.df_list]
        init_price.append(1)
        init_price = np.array(init_price, dtype=np.float64)

        self.prev_buyNhold_price = init_price
        self.init_buyNhold_amount = (INITIAL_ACCOUNT_BALANCE/len(init_price)) / init_price
        self.buyNhold_balance = INITIAL_ACCOUNT_BALANCE

        return self._next_observation()

    def render(self, mode='human', close=False, afterStep=True):
        '''
        afterStep: if is rendered after the step function, the current_step should -=1.
        '''
        todayDate = self.current_date
        if afterStep:
            todayDate -= pd.Timedelta(days = 1)

        if mode == 'human':
            # Render the environment to the screen
            profit = self.total_net_worth - INITIAL_ACCOUNT_BALANCE

            print(f'Date: {todayDate}')
            print(f'Balance: {self.cash}')
            print(
                f'Shares held: {self.shares_held} (Total sold: {self.total_shares_sold})')
            print(
                f'Avg cost for held shares: {self.cost_basis} (Total sales value: {self.total_sales_value})')
            print(
                f'Net worth: {self.net_worth} (Max net worth: {self.max_net_worth})')
            print(f'Profit: {profit}')

        elif mode == 'detail':  # Want to add all transaction details
            net_worth_delta = self.total_net_worth - self.prev_total_net_worth
            buyNhold_delta = self.buyNhold_balance - self.prev_buyNhold_balance
            return {
                "date": todayDate,
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
