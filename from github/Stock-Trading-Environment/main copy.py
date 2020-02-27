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


SAVE_DIR = "./output/100"
common_fileName_prefix = "sp500+dax+hk-Training"
summary_fileName_suffix = "summary-X.out"
detail_fileName_suffix = "detailed-ModelNo-X.out"

summary_fileName_model = common_fileName_prefix+'_'+summary_fileName_suffix
detail_fileName_model = common_fileName_prefix+'_'+detail_fileName_suffix

trainYears = 10
testYears = 5

df_namelist = ['^GSPC.csv', '^GDAXI.csv', '^HSI.csv']
rootDir = "./data"
train_df_list = []
test_df_list = []
trainStartDate = pd.to_datetime("2004-01-01")
trainEndDate = pd.to_datetime("2013-12-31")

testStartDate = pd.to_datetime("2015-01-01")
testEndDate = pd.to_datetime("2019-12-31")

for name in df_namelist:
    df = csv2df(rootDir, name, source = "yahoo")
    df = df.sort_values('Date').dropna()
    df = df.reset_index(drop=True)

    train_df = df[(df['Date'] >= trainStartDate) & (df['Date'] <= trainEndDate)]
    test_df = df[(df['Date'] >= testStartDate) & (df['Date'] <= testEndDate)]

    train_df_list.append(train_df)
    test_df_list.append(test_df)


# print (test_df)

# The algorithms require a vectorized environment to run
trainEnv = DummyVecEnv([lambda: StockTradingEnv(train_df_list,  isTraining=True)])
testEnv = DummyVecEnv([lambda: StockTradingEnv(test_df_list, isTraining=False)])


# ============ Number of days trained =============
REPEAT_NO = 10
tstep_list = [50000, 100000]
# tstep_list = [10000]

for tstep in tstep_list:
    final_result = []
    summary_fileName = summary_fileName_model[:-5] +str(tstep) + ".out"
    for modelNo in range(REPEAT_NO):
        profit_list = []
        act_profit_list = []
        detail_list = []
        model = PPO2(MlpPolicy, trainEnv, verbose=1)
        model.learn(total_timesteps=tstep, log_interval=32)
        # model.learn(total_timesteps=tstep)


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
            detail_list.append(singleDay_record)

            if testNo%365 == 0:
                print("\n============= TESTING "+str(testNo)+" =============\n")
                testEnv.render()


            

        detail_fileName = detail_fileName_model[:-5] +str(tstep) + '-' +str(modelNo) + detail_fileName_model[-4:]
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

        
