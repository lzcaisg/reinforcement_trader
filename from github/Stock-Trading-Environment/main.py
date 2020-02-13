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
import pprint
from os import path


SAVE_DIR = "./output/11/"
common_fileName_prefix = "sp500_3dim_10k-Training"
summary_fileName_suffix = "summary-13FEB.out"
detail_fileName_suffix = "detailed-ModelNo-X.out"

summary_fileName = common_fileName_prefix+'_'+summary_fileName_suffix
detail_fileName_model = common_fileName_prefix+'_'+detail_fileName_suffix

# df = pd.read_csv('./data/AAPL.csv')
# df = pd.read_csv('../../input/2006-2019/S&P 500 Historical Data.csv')

# df = csv2df('../../input/2006-2019', 'S&P 500 Historical Data.csv')
df = csv2df('../../', '^GSPC.csv', source = "yahoo")


df = df.sort_values('Date')
df = df.reset_index(drop=True)

trainYears = 10
testYears = 5

trainStartDate = pd.to_datetime("2004-01-01")
testEndDate = df['Date'].max()
testStartDate = testEndDate - pd.Timedelta(days=testYears*365)
trainEndDate = min(testStartDate-pd.Timedelta(days=1), trainStartDate+pd.Timedelta(days=trainYears*365))

print(trainStartDate, trainEndDate, testStartDate, testEndDate)

train_df = df[(df['Date'] >= trainStartDate) & (df['Date'] <= trainEndDate)]
test_df = df[(df['Date'] >= testStartDate) & (df['Date'] <= testEndDate)]


# print (test_df)

# The algorithms require a vectorized environment to run
trainEnv = DummyVecEnv([lambda: StockTradingEnv(train_df, isTraining=True)])
testEnv = DummyVecEnv([lambda: StockTradingEnv(test_df, isTraining=False)])
final_result = []

# ============ Number of days trained =============
REPEAT_NO = 10
tstep = 100000
# tstep = 100


for modelNo in range(REPEAT_NO):
    profit_list = []
    act_profit_list = []
    detail_list = []
    model = PPO2(MlpPolicy, trainEnv, verbose=1)
    model.learn(total_timesteps=tstep, log_interval=32)

    obs = testEnv.reset()

    # Test for consecutive 2000 days
    for testNo in range(365*5):
        action, _states = model.predict(obs)
        obs, rewards, done, info = testEnv.step(action)
        if done:
            print("Done")
            break
        profit_list.append(info[0]['profit'])
        act_profit_list.append(info[0]['actual_profit'])
        singleDay_record = testEnv.render(mode="detail")
        singleDay_record['testNo'] = testNo
        if singleDay_record['action'] != 0:
            detail_list.append(singleDay_record)

        if testNo%365 == 0:
            print("\n============= TESTING "+str(testNo)+" =============\n")
            testEnv.render()

    detail_fileName = detail_fileName_model[:-5] + str(modelNo) + detail_fileName_model[-4:]
    pickle.dump(detail_list, open(path.join(SAVE_DIR, detail_fileName), "wb"))

    final_result.append({
        "trainStart": trainStartDate,
        "trainEnd": trainEndDate,
        "testStart": testStartDate,
        "testEnd": testEndDate,
        "train_step": tstep,
        "mean": np.mean(profit_list),
        "max": np.max(profit_list),
        "min": np.min(profit_list),
        "std": np.std(profit_list),
        "final": profit_list[-1],
        "act_mean": np.mean(act_profit_list),
        "act_max": np.max(act_profit_list),
        "act_min": np.min(act_profit_list),
        "act_std": np.std(act_profit_list),
        "act_final": act_profit_list[-1],
        "total_shares_sold": info[0]['total_shares_sold']
    })
    pickle.dump(final_result, open(path.join(SAVE_DIR, summary_fileName), "wb"))
    print("********* LENTH: ", len(final_result), " *********")
    pprint.pprint (final_result[-1])

        
