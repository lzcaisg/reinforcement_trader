import gym
import json
import datetime as dt
import pickle
import ta

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

from env.FinalEnv import RebalancingEnv, DEFAULT_PARAMETER

import pandas as pd
import numpy as np
from CSVUtils import csv2df
import pprint
import os
from os import path

def main(   TRAINING = True, SAVE_DIR = "./output/1000", DATE_PREFIX = "0418", VAIRABLE_PREFIX = "action_frequency", 
            DF_NAMELIST=None, TRAIN_TEST_DATE=None, TRAIN_FOREX_ADJUST=True, TSTEP_LIST = [200000], 
            LOAD_DIR = "./output/306", MODEL_FILE_PREFIX = "BRZ_TW_NASDAQ-Selected_Trans-withleakage+RSI-200000-",
            ENV_PARAM=None):
    
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    
    # Handle File Names
    common_fileName_prefix = DATE_PREFIX+"-"+VAIRABLE_PREFIX+"-"
    summary_fileName_suffix = "summary-X.out"
    detail_fileName_suffix = "detailed-ModelNo_X.out"

    summary_fileName_model = common_fileName_prefix+'-'+summary_fileName_suffix
    detail_fileName_model = common_fileName_prefix+'-'+detail_fileName_suffix

    
    if TRAIN_TEST_DATE is None:
        trainStartDate  = pd.to_datetime("2005-01-01")
        trainEndDate    = pd.to_datetime("2014-12-31")

        testStartDate   = pd.to_datetime("2015-01-01")
        testEndDate     = pd.to_datetime("2019-12-31")
    else:
        trainStartDate  = TRAIN_TEST_DATE['trainStartDate']
        trainEndDate    = TRAIN_TEST_DATE['trainEndDate']

        testStartDate   = TRAIN_TEST_DATE['testStartDate']
        testEndDate     = TRAIN_TEST_DATE['testEndDate']

    if DF_NAMELIST is None:
        df_namelist = {"high": "^BVSP_new", "mid": "^TWII_new", "low": "^IXIC_new"}
    else:
        df_namelist = DF_NAMELIST

    rootDir = "./data"
    train_df_dict   = {"high": pd.DataFrame(), "mid": pd.DataFrame(), "low": pd.DataFrame()}
    test_df_dict    = {"high": pd.DataFrame(), "mid": pd.DataFrame(), "low": pd.DataFrame()}


    for key in df_namelist:
        fileName = df_namelist[key]+".csv"
        
        source, price_label = set_source_label(df_namelist[key], TRAIN_FOREX_ADJUST)
        
        df = csv2df(rootDir, fileName, source = source)
        df = get_ta(df, price_label)

        train_df = df[(df['Date'] >= trainStartDate) & (df['Date'] <= trainEndDate)]
        test_df  = df[(df['Date'] >= testStartDate)  & (df['Date'] <= testEndDate)]

        train_df_dict[key] = train_df
        test_df_dict[key]  = test_df

    col_list = ['EMA', 'MACD_diff', 'delta_time', 'RSI', 'Cum FX Change']
    
    if ENV_PARAM is None:
        ENV_PARAM = {}
    # Set Default Values
    for key in DEFAULT_PARAMETER:
        if (not key in ENV_PARAM.keys()) or (not type(ENV_PARAM[key]) is type(DEFAULT_PARAMETER[key])):
            ENV_PARAM[key] = DEFAULT_PARAMETER[key]

    if TRAINING:
        trainEnv = DummyVecEnv([lambda: RebalancingEnv(df_dict=train_df_dict, col_list=col_list, isTraining=True, env_param=ENV_PARAM)])
    testEnv  = DummyVecEnv([lambda: RebalancingEnv(df_dict=test_df_dict, col_list=col_list, isTraining=False, env_param=ENV_PARAM)])


    # ============ Number of days trained =============
    REPEAT_NO = 10
    tstep_list = TSTEP_LIST

    for tstep in tstep_list:
        final_result = []
        summary_fileName = summary_fileName_model[:-5] +str(tstep) + ".out"
        for modelNo in range(REPEAT_NO):
            if TRAINING:
                print("\n============= START TRAINING "+str(modelNo)+" =============\n")
                
                model = PPO2(MlpPolicy, trainEnv, verbose=1)
                # model = PPO2(MlpPolicy, trainEnv, verbose=1, tensorboard_log="./"+SAVE_DIR[-3:]+'_'+str(tstep)+"_tensorboard/")
                model.learn(total_timesteps=tstep, log_interval=128)
                model_name = common_fileName_prefix + str(tstep) + '-' +str(modelNo) + "-model.model"
                model.save(path.join(SAVE_DIR, model_name), cloudpickle=True)
            else:
                model_name = MODEL_FILE_PREFIX+str(modelNo) + "-model.model"
                model = PPO2.load(path.join(LOAD_DIR, model_name))

            profit_list = []
            act_profit_list = []
            detail_list = []
            print("\n============= START TESTING "+str(modelNo)+" =============\n")
            obs = testEnv.reset()
            for testNo in range((testEndDate-testStartDate).days):      # Set index number of date as TestNo
                action, _states = model.predict(obs)
                if np.isnan(action).any():
                    print(testNo)
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
                    print("\n======= TESTING "+str(testNo)+" =======\n")
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


def set_source_label(file_name, TRAIN_FOREX_ADJUST):
    if file_name[-4:]=="_new":
        source = "done"
        if TRAIN_FOREX_ADJUST:
            price_label = 'Actual Price'
        else:
            price_label = 'Price'
    else:
        source = "yahoo"
        price_label = 'Price'
    return source, price_label

        
def get_ta(df, price_label):
    df = df.sort_values('Date').reset_index(drop=True)
    # Add TA Indicators
    # <1> EMA
    df['EMA'] = df[price_label].ewm(span=15).mean()
    df['EMA'] /= df[price_label][0]

    # <2> MACD_diff
    df['MACD_diff'] = ta.trend.macd_diff(df[price_label], fillna=True)
    df['EMA'] /= df[price_label][0]
    macd_direction = df['MACD_diff']/np.abs(df['MACD_diff']) # 1: No change, -1: Change Sign

    # <3> MACD_change
    df['MACD_change'] = (-1*macd_direction*macd_direction.shift(1)+1)/2 # 1: Change Sign, 0: No Change

    # <4> delta_time: How many days since the last trend change
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

    # <5> RSI
    df['RSI'] = ta.momentum.RSIIndicator(df[price_label], fillna=True).rsi() 
    df['RSI'] /= df[price_label][0]

    roll_Max = df[price_label].rolling(window=30, min_periods=1).max()
    df['daily_Drawdown'] = df[price_label]/roll_Max - 1.0

    # Clean up all the nans
    df = df.dropna()
    df = df.reset_index(drop=True)
    return df
