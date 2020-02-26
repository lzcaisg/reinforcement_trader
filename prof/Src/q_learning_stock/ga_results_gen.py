# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
from pathlib import Path
try:
    os.chdir(os.path.join(os.getcwd(), 'eddy_src/q_learning_stock'))
    print(os.getcwd())
except:
    pass

import indicators
import util
import config
import numpy as np
import pandas as pd
import time
import calendar
import math
import util

#%% [markdown]
# # Instructions for running:
# ### 1. Choose to run goldman or index portfolio in choose_set
# ### 3. Set save_algo_data to True or False. Save the calculated created files
# ### 2. Set save_passive to True or False. It is the daily passive NAV (net asset value) gain (buy and hold)
# ### 4. Adjust portfolio composition parameters as needed in get_algo_results in algo_dataset (should use values from the genetic algorithm)
# ### 5. Run all cells


#%%
# Do not change run_set order. The order is hardcoded into below code
run_set = ['goldman', 'index', '^BVSP', '^TWII', '^IXIC', 'index_sampled']
choose_set_num = 5
save_algo_data = True
save_passive = False
base_rate = False

#%%
if base_rate:
    util.gen_algo_data(run_set, choose_set_num, save_algo_data, save_passive)
else:
    util.gen_algo_data(run_set, choose_set_num, save_algo_data, save_passive, save_sub_folder='no_br/', base_rates=[0,0,0])

#%%
