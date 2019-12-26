import CSVUtils as csvutils
import pandas as pd
import numpy as np
import os
import pickle

DIR_FILENAME = "../index"
NAMELIST_FILENAME = "Index-v3-2009&1990.csv"
DIR_DATA_1990 = "1990-2005"
DIR_DATA_2006 = "2006-2019"
INIT_VALUE = 1000000
START_DATE = pd.Timestamp(2009, 1, 1)
END_DATE = pd.Timestamp(2018, 12, 31)
namelist_df = pd.read_csv(os.path.join(DIR_FILENAME, NAMELIST_FILENAME))

for etfName in namelist_df['File Name']:
# for etfName in ['Bursatil']: #------For Testing One ETF------
# for etfName in [namelist_df['File Name'][0]]: #------For Testing Input------
    print(etfName)

    # etfName = "S&P Merval"
    fileName = etfName + " Historical Data.csv"
    dfList = []

    try:
        etfDF = csvutils.csv2df(os.path.join(DIR_FILENAME, DIR_DATA_2006), fileName).sort_values(by=['Date']).reset_index(drop=True)

        dateRange = pd.date_range(START_DATE, END_DATE)
        etfDF.set_index('Date', inplace=True)
        etfDF = etfDF.reindex(dateRange, fill_value=np.nan)
        priceArray = np.array(etfDF['Price'].reset_index(drop=True))
        resultMatrix = np.triu(np.outer(1/priceArray, priceArray))
        '''
        outer(a, b) = [
            [a0*b0, a0*b1, ..., a0*bn],
            [a1*b0, a1*b1, ..., a1*bn],
            ...
            [an*b0, an*b1, ..., an*bn]
        ];
        
        Therefore, outer(1/p, p) = 
        [
            [p0/p0, p1/p0, ..., pn/p0],
            [p0/p1, p1/p1, ..., pn/p1],
            ...
            [p0/pn, p1/pn, ..., pn/pn]
        ]
        
        triu(outer(1/p, p)) = 
        [
            [p0/p0, p1/p0, ..., pn/p0],
            [    0, p1/p1, ..., pn/p1],
            ...
            [    0,     0, ..., pn/pn]
        ]
        
        Therefore, the return for buying in at time t and sell at time s=t+n days will be 
        (ps/pt) = result[t][s] = result[t][t+n]
        
        '''
        pickle.dump(resultMatrix, open(os.path.join("output", etfName + "_returnMatrix.out"), "wb"))
    except Exception as e:
        print(e)