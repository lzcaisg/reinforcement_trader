import CSVUtils as csvutils
import pandas as pd
import numpy as np
import os
import pickle
from numba import jit

DIR_FILENAME = "../index"
NAMELIST_FILENAME = "Index-v3-2009&1990.csv"
DIR_DATA_1990 = "1990-2005"
DIR_DATA_2006 = "2006-2019"
INIT_VALUE = 1000000
START_DATE = pd.Timestamp(2009, 1, 1)
namelist_df = pd.read_csv(os.path.join(DIR_FILENAME, NAMELIST_FILENAME))
# print([namelist_df['File Name'][0]])

# for etfName in namelist_df['File Name']:
for etfName in [namelist_df['File Name'][0]]:
    print(etfName)

    # etfName = "S&P Merval"
    fileName = etfName + " Historical Data.csv"
    dfList = []
    # for dirName in [DIR_DATA_1990, DIR_DATA_2006]:
    #     dfList.append(csvutils.csv2df(os.path.join(DIR_FILENAME, dirName), fileName))

    # longdf = pd.concat(dfList).sort_values(by=['Date']).reset_index(drop=True)

    longdf = csvutils.csv2df(os.path.join(DIR_FILENAME, DIR_DATA_2006), fileName).sort_values(by=['Date']).reset_index(drop=True)
    dateList = list(longdf['Date'])
    dateList = [i for i in dateList if i >= START_DATE]
    deltaList = list(range(7, 365*5))
    resultdf = pd.DataFrame(index=dateList, columns=deltaList)
    longdf.set_index('Date', inplace=True)
    latestDate = dateList[-1]
    counter = 0
    for currentDay in dateList:
        for deltaday in range(7, 365 * 5):
            sellDate = currentDay + pd.Timedelta(days=deltaday)
            if sellDate > latestDate:
                break
            if not ((currentDay in dateList) and (sellDate in dateList)):
                continue

            buyPrice = longdf['Price'].loc[currentDay]
            sellPrice = longdf['Price'].loc[currentDay + pd.Timedelta(days=deltaday)]
            resultdf.at[currentDay, deltaday] = sellPrice / buyPrice
        if counter % 50 == 0:
            print(etfName+" "+str(currentDay) + " finished.")
        counter += 1
    print(resultdf)
    pickle.dump(resultdf, open(os.path.join("output", etfName+"_return.out"), "wb"))

