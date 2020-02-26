# General
verbose = 0
grid_search = {
    "batch_size": [32, 64, 128],
    "time_steps": [30, 60, 90],
    "lr": [0.01, 0.001, 0.0001],
    "epochs": [30, 50, 70]
}
log_time_format = '%Y%m%d_%H%M%S'

# Training hyper parameters
batch_size = 64
num_epochs = 10
learning_rate = 0.001
## LSTM
time_steps = 60
## Reinforcement learning
discount_rate = .95             # discount is related to future reward value
exploration_rate = 0.1
mem_len = 1000 # max length of memory
gamma = 0.95
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995

# Stock settings
trade_period = 5
commission_rate = 0.002

# Data
start_date = '2016-04-28'
end_date = '2017-04-28'
dataset_path = r'data/jpm.csv'
column_names = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
feature_names = ['Date', 'Open', 'High', 'Low', 'Volume']
label_name = 'Close'
train_split = 0.8               # Less than 1

epoch_print_number = 1

# Do not change (unless you know what you are doing!)
actions = ['hold', 'buy', 'sell']
num_actions = len(actions)

# Notes
# Key terms: Agent, Action, Discount, Environment, State, Reward, Policy, Value, Q-value, Trajectory
# q-value: Q maps state-action pairs to rewards.
# Qπ(s, a) refers to the long-term return of the current state s, taking action a under policy π