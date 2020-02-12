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

    def __init__(self, df, isTraining=True):
        super(StockTradingEnv, self).__init__()

        self.training = isTraining
        self.window_size = 5

        self.df = df.reset_index(drop=True)
        # self.reward_range = (0, MAX_ACCOUNT_BALANCE)  # Legacy, Deleted on 10/FEB, we want negative

        # Actions of the format Buy x%, Sell x%, Hold, etc.
        self.action_space = spaces.Box(
            low=np.array([0, 0]), high=np.array([3, 1]), dtype=np.float16)

        # Prices contains the OHCL values for the last five prices
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(6, 6), dtype=np.float16)

    def _next_observation(self):
        '''
        The _next_observation method compiles the stock data for the last five time steps, 
        appends the agent’s account information, and scales all the values to between 0 and 1.
        '''
        
        # self.current_step is defined in reset method,
        # We assume the current_step is TODAY (BEFORE FINAL), which means we only know infomation till YESTERDAY ENDS.

        frame = np.array([
            self.df.loc[self.current_step-self.window_size : self.current_step, # Not including current_step(TODAY)
                'Open'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step-self.window_size : self.current_step, 
                'High'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step-self.window_size : self.current_step, 
                'Low'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step-self.window_size : self.current_step, 
                'Price'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step-self.window_size : self.current_step, 
                'Vol'].values / MAX_NUM_SHARES,
        ])


        # Append additional data and scale each value to between 0-1
        obs = np.append(frame, [[
            self.balance / MAX_ACCOUNT_BALANCE,
            self.max_net_worth / MAX_ACCOUNT_BALANCE,
            self.shares_held / MAX_NUM_SHARES,
            self.cost_basis / MAX_SHARE_PRICE,
            self.total_shares_sold / MAX_NUM_SHARES,
            self.total_sales_value / (MAX_NUM_SHARES * MAX_SHARE_PRICE),
        ]], axis=0)


        return obs

    def _take_action(self, action):
        # Set the current price to a random price within the time step
        self.actual_price = random.uniform(
            self.df.loc[self.current_step, "Low"], self.df.loc[self.current_step, "High"])

        action_type = action[0]
        amount = action[1]

        if action_type < 1:
            # Buy amount % of balance in shares
            

            total_possible = int(self.balance / self.actual_price)
            shares_bought = int(total_possible * amount)
            self.current_action = shares_bought
            prev_cost = self.cost_basis * self.shares_held
            additional_cost = shares_bought * self.actual_price * (1+COMMISSION_FEE)

            self.balance -= additional_cost
            self.cost_basis = (
                prev_cost + additional_cost) / (self.shares_held + shares_bought)
            self.shares_held += shares_bought

        elif action_type < 2:
            # Sell amount % of shares held

            shares_sold = int(self.shares_held * amount)
            self.current_action = shares_sold*-1
            self.balance += shares_sold * self.actual_price
            self.shares_held -= shares_sold
            self.total_shares_sold += shares_sold
            self.total_sales_value += shares_sold * self.actual_price

        self.prev_net_worth = self.net_worth
        self.net_worth = self.balance + self.shares_held * self.actual_price

        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth

        if self.shares_held == 0:
            self.cost_basis = 0

    def step(self, action):
        '''
        Next, our environment needs to be able to take a step. 
        At each step we will take the specified action (chosen by our model), 
        calculate the reward, and return the next observation.
        '''

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

        finished = False
        
        # 2. Determine TOMORROW's Date (For training)
        if self.current_step > len(self.df.loc[:, 'Open'].values) - 1:
            # if self.training:
            if self.training:
                self.current_step = self.window_size  # Going back to time 0
            else:
                self.current_step -= 1
                finished = True

        '''
        We want to incentivize profit that is sustained over long periods of time. 
        At each step, we will set the reward to the account balance multiplied by 
        some fraction of the number of time steps so far.

        The purpose of this is to delay rewarding the agent too fast in the early stages 
        and allow it to explore sufficiently before optimizing a single strategy too deeply. 
        It will also reward agents that maintain a higher balance for longer, 
        rather than those who rapidly gain money using unsustainable strategies.
        '''
        self.init_buyNhold_balance = self.init_buyNhold_amount * self.df.loc[self.current_step-1, "Price"]
        
        profit = self.net_worth - INITIAL_ACCOUNT_BALANCE
        actual_profit = self.net_worth - self.init_buyNhold_balance
        
        delay_modifier = (self.current_step / MAX_STEPS)
        # reward = self.balance * delay_modifier  # Original Version
        reward = actual_profit * delay_modifier  # Use Actual Net Profit

        done = (self.net_worth <= 0) or finished
        obs = self._next_observation()

        info = {"profit": profit, "total_shares_sold": self.total_shares_sold, "actual_profit": actual_profit}

        return obs, reward, done, info

    def reset(self):
        # Reset the state of the environment to an initial state
        self.balance = INITIAL_ACCOUNT_BALANCE
        self.net_worth = INITIAL_ACCOUNT_BALANCE
        self.max_net_worth = INITIAL_ACCOUNT_BALANCE
        self.shares_held = 0
        self.cost_basis = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0
        self.current_action = 0
        self.prev_net_worth = INITIAL_ACCOUNT_BALANCE
        

        # Set the current step to a random point within the data frame
        # We set the current step to a random point within the data frame, because it essentially gives our agent’s more unique experiences from the same data set.
        if self.training:
            self.current_step = random.randint(
                self.window_size, len(self.df.loc[:, 'Open'].values))
        else:
            self.current_step = self.window_size
        
        self.init_buyNhold_amount = INITIAL_ACCOUNT_BALANCE/self.df.loc[self.current_step, "Price"]
        self.init_buyNhold_balance = INITIAL_ACCOUNT_BALANCE

        return self._next_observation()

    def render(self, mode='human', close=False, afterStep=True):
        '''
        afterStep: if is rendered after the step function, the current_step should -=1.
        '''
        todayStep = self.current_step
        if afterStep:
            todayStep -= 1
            
        if mode=='human':
            # Render the environment to the screen
            profit = self.net_worth - INITIAL_ACCOUNT_BALANCE

            print(f'Step: {todayStep}')
            print(f'Balance: {self.balance}')
            print(
                f'Shares held: {self.shares_held} (Total sold: {self.total_shares_sold})')
            print(
                f'Avg cost for held shares: {self.cost_basis} (Total sales value: {self.total_sales_value})')
            print(
                f'Net worth: {self.net_worth} (Max net worth: {self.max_net_worth})')
            print(f'Profit: {profit}')
        
        elif mode=='detail': # Want to add all transaction details
            return {
                "step": todayStep,
                "date": self.df.loc[todayStep, "Date"],
                "actual_price": self.actual_price,
                "action": self.current_action,
                "shares_held": self.shares_held,
                "net_worth": self.net_worth,
                "net_worth_delta": self.net_worth - self.prev_net_worth,
                "buyNhold_balance": self.init_buyNhold_balance,
                "actual_profit": self.net_worth - self.init_buyNhold_balance,
            }

