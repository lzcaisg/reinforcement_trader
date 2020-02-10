# Class Description for the Environment

## 0. Constant:
- MAX_ACCOUNT_BALANCE = 2147483647: Denominator
- MAX_NUM_SHARES = 2147483647:      Denominator
- MAX_SHARE_PRICE = 5000:           Denominator


- MAX_OPEN_POSITIONS = 5:           Not in use    
- MAX_STEPS = 20000:                Denominator
- INITIAL_ACCOUNT_BALANCE = 10000

## 1. Original Attributes
- self.isTraining (boolean): Whether this class is used for training:
    - If so, the environment will go back to the start date when reaching to the end;
    - If not, the environment will stop when reaching the end.

- self.window_size (int): Window size for the observation, # of days of data returned to the agent

- self.df (pd.DataFrame): Import dataframe. Should be in the format of 
    - ["Date", "Open", "High", "Low", "Price", "Vol", "Change"]

- self.action_space (Gym.spaces.Box): NOT CHECKED YET, used as default

- self.observation_space (Gym.spaces.Box): NOT CHECKED YET, used as default

## 2. Calculated and Status Attributed (For One Stock Only)
- self.current_step: The latest unobserved record. Suppose self.current_step = d
    - The observation will provide records from [d-window_size, ..., d-1], i.e. total number equals window_size. (END of) today's data will not be provided to prevent leeking into future.
    - The execution price is randomly selected between the open price and the close price of the day d.
    - Need to change to date when multiple stocks are used.


- self.net_worth = self.balance + self.shares_held * current_price
    - Value of Cash + Stock

- self.balance: Total CASH on hand

- self.max_net_worth: Max Net Worth (Cash + Stock)

- self.shares_held: Current number of stocks hold

- self.cost_basis: The original value or purchase price of the stocks. i.e. The history cost of the stock. Updated as a weighted average inventory.

- self.total_shares_sold: Total number of stocks sold

- self.total_sales_value: Total cashed value from selling the stocks

- self.init_buyNhold_amount: If we all-in to buy the stock in the first day, what is the amount we can buy.

- self.init_buyNhold_balance: If we all-in to buy the stock in the first day, what is the current value of the stock.



