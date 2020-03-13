import gym
import json
import datetime as dt
import pickle
import ta

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

from env.MonthlyRebalancingEnv import RebalancingEnv

import pandas as pd
import numpy as np
from CSVUtils import csv2df
import pprint
import os
from os import path


SAVE_DIR = "./output/303"
import os
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

common_fileName_prefix = "BRZ_TW_NASDAQ-Selected_Trans-7d"
summary_fileName_suffix = "summary-X.out"
detail_fileName_suffix = "detailed-ModelNo-X.out"

summary_fileName_model = common_fileName_prefix+'_'+summary_fileName_suffix
detail_fileName_model = common_fileName_prefix+'_'+detail_fileName_suffix

trainYears = 10
testYears = 5


df_namelist = {"high": "^BVSP_new", "mid": "^TWII_new", "low": "^IXIC_new"}
# df_namelist = {"high": "^TWII", "mid": "^IXIC", "low": "^BVSP"}

#
rootDir = "./data"
train_df_dict   = {"high": pd.DataFrame(), "mid": pd.DataFrame(), "low": pd.DataFrame()}
test_df_dict    = {"high": pd.DataFrame(), "mid": pd.DataFrame(), "low": pd.DataFrame()}

trainStartDate  = pd.to_datetime("2005-01-01")
trainEndDate    = pd.to_datetime("2014-12-31")

testStartDate   = pd.to_datetime("2015-01-01")
testEndDate     = pd.to_datetime("2019-12-31")

for key in df_namelist:
    fileName = df_namelist[key]+".csv"
    if df_namelist[key][-4:]=="_new":
        source = "done"
        # price_label = 'Actual Price'
        price_label = 'Price'

    else:
        source = "yahoo"
        price_label = 'Price'
    df = csv2df(rootDir, fileName, source = source)
    

    df = df.sort_values('Date').reset_index(drop=True)

    df['EMA'] = df[price_label].ewm(span=15).mean()
    df['MACD_diff'] = ta.trend.macd_diff(df[price_label])
    macd_direction = df['MACD_diff']/np.abs(df['MACD_diff']) # 1: No change, -1: Change Sign
    df['MACD_change'] = (-1*macd_direction*macd_direction.shift(1)+1)/2 # 1: Change Sign, 0: No Change

    # delta_time: How many days since the last trend change
    delta_time = [] 
    for i in df['MACD_change']:
        if len(delta_time) == 0:
            result = 0
        elif i==0:
            result = delta_time[-1]+1
        else: #Nan or 1
            result = 0
        delta_time.append(result)
    df['delta_time'] = delta_time

    # Clean up all the nans
    df = df.dropna()
    df = df.reset_index(drop=True)

    train_df = df[(df['Date'] >= trainStartDate) & (df['Date'] <= trainEndDate)]
    test_df  = df[(df['Date'] >= testStartDate)  & (df['Date'] <= testEndDate)]

    train_df_dict[key] = train_df
    test_df_dict[key]  = test_df

col_list = ['EMA', 'MACD_diff', 'delta_time']
# The algorithms require a vectorized environment to run
trainEnv = DummyVecEnv([lambda: RebalancingEnv(df_dict=train_df_dict, col_list=col_list, isTraining=True)])
testEnv  = DummyVecEnv([lambda: RebalancingEnv(df_dict=test_df_dict, col_list=col_list, isTraining=False)])


# ============ Number of days trained =============
REPEAT_NO = 10
# tstep_list = [200000,500000]
# tstep_list = [500000, 1000000]
tstep_list = [100000]
# tstep_list = [100000, 500000]


for tstep in tstep_list:
    final_result = []
    summary_fileName = summary_fileName_model[:-5] +str(tstep) + ".out"
    for modelNo in range(REPEAT_NO):
        profit_list = []
        act_profit_list = []
        detail_list = []
        model = PPO2(MlpPolicy, trainEnv, verbose=1, tensorboard_log="/0313_"+str(tstep)+"_tensorboard/")
        model.learn(total_timesteps=tstep, log_interval=128)
        # model.learn(total_timesteps=tstep)
        model_name = common_fileName_prefix + str(tstep) + '-' +str(modelNo) + "-model.model"
        model.save(path.join(SAVE_DIR, model_name), cloudpickle=True)


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
            singleDay_record['rewards'] = rewards[0]
            detail_list.append(singleDay_record)

            if testNo%365 == 0:
                print("\n============= TESTING "+str(testNo)+" =============\n")
                testEnv.render()


            

        detail_fileName = detail_fileName_model[:-5] + str(tstep) + '-' +str(modelNo) + detail_fileName_model[-4:]
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

        
