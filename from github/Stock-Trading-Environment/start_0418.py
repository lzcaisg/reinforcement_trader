import pandas as pd
import numpy as np
from main_0418 import *

SAVE_DIR = "./output/1001"                              # Save Directory
DATE_PREFIX = "0418"            
VAIRABLE_PREFIX = "monthlyFreq_haveLeak_noCrisis"           # Filename: "0418-action_frequency-summary.out"
TSTEP_LIST = [200000]                                   # List of training steps
DF_NAMELIST = {"high": "^BVSP_new", "mid": "^TWII_new", "low": "^IXIC_new"}
                                                        # csv file name

freq_dict = {
    "daily": 1,
    "weekly": 5,
    "biweekly": 10,
    "monthly": 20,
}

ENV_PARAM = {
    "trans_freq": freq_dict['monthly'],
    "have_currency_leakage": True,
    "crisis_detection": True,
    "MDD_window": freq_dict['monthly'], 
    "reward_wait": 10,
    "MDD_threshold": 0.2
}

TRAIN_TEST_DATE = {                     # outer bond of the train test date
    "trainStartDate": pd.to_datetime("2005-01-01"),
    "trainEndDate":   pd.to_datetime("2014-12-31"),
    "testStartDate":  pd.to_datetime("2000-01-01"),
    "testEndDate":    pd.to_datetime("2008-12-31")
    }


# --- To Train a new environment --- #
# main(   TRAINING=True, SAVE_DIR = SAVE_DIR, DATE_PREFIX = DATE_PREFIX, 
#         VAIRABLE_PREFIX = VAIRABLE_PREFIX, DF_NAMELIST = DF_NAMELIST, 
#         TRAIN_TEST_DATE = TRAIN_TEST_DATE, TSTEP_LIST = TSTEP_LIST,
#         ENV_PARAM = ENV_PARAM)


# # --- To Test an existing environment --- #
dir_dict = {
    "monthly": "1011",
    "biweekly": "1012",
    "weekly": "1013"
}

threshold_list = [0.15, 0.10]
task_list = []
LOAD_DIR = "./output/306"
MODEL_FILE_PREFIX = "BRZ_TW_NASDAQ-Selected_Trans-withleakage+RSI-200000-"

# for thres in threshold_list:
#     for freq in dir_dict:
#         SAVE_DIR = "./output/"+dir_dict[freq]
#         ENV_PARAM['SAVE_DIR'] = SAVE_DIR
#         for start in range(2000, 2016):
#             testStartDate = pd.to_datetime(str(start)+"-01-01")
#             ENV_PARAM['testStartDate'] = testStartDate
#             for end in range(start+4, 2020):
#                 testEndDate = pd.to_datetime(str(end)+ "-12-31")
#                 ENV_PARAM['VAIRABLE_PREFIX'] = "TEST_"+("%.2f"%thres)+"_"+freq+"Crisis_"+str(start)+"_"+str(end)
#                 ENV_PARAM['MDD_window'] = freq_dict[freq]
#                 ENV_PARAM['MDD_threshold'] = thres

#                 task_list.append(ENV_PARAM)
    
manytest(   freq_dict=freq_dict, dir_dict=dir_dict, threshold_list=threshold_list,
            startyear=2000, endyear=2019, SAVE_DIR_PREFIX="./output/",
            LOAD_DIR = LOAD_DIR, MODEL_FILE_PREFIX = MODEL_FILE_PREFIX )




# ----------- Below for Testing Only ------------------
# TRAIN_TEST_DATE['testStartDate'] = testStartDate
# TRAIN_TEST_DATE['testEndDate'] = testEndDate
# thres = 0.2
# freq="monthly"
# start = 2000
# end = 2008
# LOAD_DIR = "./output/306"
# MODEL_FILE_PREFIX = "BRZ_TW_NASDAQ-Selected_Trans-withleakage+RSI-200000-"

# VAIRABLE_PREFIX = "TEST_"+("%.2f"%thres)+"_"+freq+"Crisis_"+str(start)+"_"+str(end)
# ENV_PARAM['MDD_window'] = freq_dict[freq]
# ENV_PARAM['MDD_threshold'] = thres
# # print("@@@@@@@@@@@@@@@@@@@", thres, freq, start, end, "@@@@@@@@@@@@@@@@@@@")
# main(   TRAINING=False, SAVE_DIR = SAVE_DIR, LOAD_DIR = LOAD_DIR, MODEL_FILE_PREFIX = MODEL_FILE_PREFIX,
#         DATE_PREFIX = DATE_PREFIX, VAIRABLE_PREFIX = VAIRABLE_PREFIX, DF_NAMELIST = DF_NAMELIST, 
#         TRAIN_TEST_DATE = TRAIN_TEST_DATE, TSTEP_LIST = TSTEP_LIST, ENV_PARAM=ENV_PARAM)