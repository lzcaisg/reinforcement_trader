import gym
import json
import datetime as dt

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

from env.StockTradingEnv import StockTradingEnv

import pandas as pd

from CSVUtils import csv2df

# df = pd.read_csv('./data/AAPL.csv')
# df = pd.read_csv('../../input/2006-2019/S&P 500 Historical Data.csv')
df = csv2df('../../input/2006-2019', 'S&P 500 Historical Data.csv')

df = df.sort_values('Date')
df = df.reset_index(drop=True)
# print (df)

# The algorithms require a vectorized environment to run
env = DummyVecEnv([lambda: StockTradingEnv(df)])

model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=20000)

obs = env.reset()
for i in range(2000):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()
