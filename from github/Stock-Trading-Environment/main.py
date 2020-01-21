import gym
import json
import datetime as dt
import pickle

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

from env.StockTradingEnv import StockTradingEnv

import pandas as pd
import numpy as np
from CSVUtils import csv2df

REPEAT_NO = 100

# df = pd.read_csv('./data/AAPL.csv')
# df = pd.read_csv('../../input/2006-2019/S&P 500 Historical Data.csv')
df = csv2df('../../input/2006-2019', 'S&P 500 Historical Data.csv')

df = df.sort_values('Date')
df = df.reset_index(drop=True)
# print (df)

# The algorithms require a vectorized environment to run
env = DummyVecEnv([lambda: StockTradingEnv(df)])
final_result = []

timestep_list = [100000]
for tstep in timestep_list:
    for i in range(REPEAT_NO):
        profit_list = []
        model = PPO2(MlpPolicy, env, verbose=1)
        model.learn(total_timesteps=tstep, log_interval=32)

        obs = env.reset()

        # Test for consecutive 2000 days
        for i in range(2000):
            action, _states = model.predict(obs)
            obs, rewards, done, info = env.step(action)
            profit_list.append(info[0]['profit'])
            
            if i%200 == 0:
                print("\n============= TESTING "+str(i)+" =============\n")
                env.render()
    

        final_result.append({
            "train_step": tstep,
            "mean": np.mean(profit_list),
            "max": np.max(profit_list),
            "min": np.min(profit_list),
            "std": np.std(profit_list),
            "final": profit_list[-1],
            "total_shares_sold": info[0]['total_shares_sold']
        })
        pickle.dump(final_result, open("./output/4/sp500_train100000_test2000_repeat100_withsold_21JAN2020-0.out", "wb"))
        print (final_result)

        
