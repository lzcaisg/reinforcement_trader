import pandas as pd
import numpy as np
from main_0418 import *

SAVE_DIR = "./output/1000"                              # Save Directory
DATE_PREFIX = "0418"            
VAIRABLE_PREFIX = "monthlyFreq_noLeak_noCrisis"           # Filename: "0418-action_frequency-summary.out"
TSTEP_LIST = [200000]                                   # List of training steps
DF_NAMELIST = {"high": "^BVSP_new", "mid": "^TWII_new", "low": "^IXIC_new"}
                                                        # csv file name

freq_dict = {
    "daily": 1,
    "weekly": 5,
    "monthly": 20,
}

ENV_PARAM = {
    "trans_freq": freq_dict['monthly'],
    "have_currency_leakage": False,
    "crisis_detection": False,
    "MDD_window": freq_dict['monthly'], 
    "reward_wait": 10
}

TRAIN_TEST_DATE = {                     # outer bond of the train test date
    "trainStartDate": pd.to_datetime("2005-01-01"),
    "trainEndDate":   pd.to_datetime("2014-12-31"),
    "testStartDate":  pd.to_datetime("2015-01-01"),
    "testEndDate":    pd.to_datetime("2019-12-31")
    }


# --- To Train a new environment --- #
main(   TRAINING=True, SAVE_DIR = SAVE_DIR, DATE_PREFIX = DATE_PREFIX, 
        VAIRABLE_PREFIX = VAIRABLE_PREFIX, DF_NAMELIST = DF_NAMELIST, 
        TRAIN_TEST_DATE = TRAIN_TEST_DATE, TSTEP_LIST = TSTEP_LIST,
        ENV_PARAM = ENV_PARAM)


# # --- To Test an existing environment --- #
# SAVE_DIR = "./output/1000"
# LOAD_DIR = "./output/306"
# MODEL_FILE_PREFIX = "BRZ_TW_NASDAQ-Selected_Trans-withleakage+RSI-200000-"
# main(   TRAINING=False, SAVE_DIR = SAVE_DIR, LOAD_DIR = LOAD_DIR, MODEL_FILE_PREFIX = MODEL_FILE_PREFIX,
#         DATE_PREFIX = DATE_PREFIX, VAIRABLE_PREFIX = VAIRABLE_PREFIX, DF_NAMELIST = DF_NAMELIST, 
#         TRAIN_TEST_DATE = TRAIN_TEST_DATE, TSTEP_LIST = TSTEP_LIST, ENV_PARAM=ENV_PARAM)
