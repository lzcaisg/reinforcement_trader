import datetime
import warnings
import pandas as pd
import numpy as np
from MongoDBUtils import *
from scipy.optimize import fsolve

TRADING_FEE = 0.008


def getOneDayRecord(date, col_name="S&P 500"):
    client, db, collection = setUpMongoDB(col_name=col_name)
    query = {"Date": date}
    result = collection.find_one(query)
    client.close()
    return result


def getRecordFromETFList(date, etfList):
    '''

    :param date:
    :param etfList:
    :return: A df like this:

                        Value
        Name
        Hang Seng	    30
        S&P 500	        40
        STI	            NaN
        Shanghai	    50

    '''
    if not isinstance(etfList, list):
        warnings.warn("Environment/getRecordFromETFList() Warning: etfList is not List")
        return None

    client, db, _ = setUpMongoDB()

    resultDF = pd.DataFrame(etfList, columns=["Name"]).set_index('Name', drop=True)
    resultDF['Value'] = np.nan

    for etf in etfList:
        if etf == "CASH":
            resultDF['Value'][etf] = 1
        else:
            collection = db[etf]
            query = {"Date": date}
            result = collection.find_one(query)
            if result:
                resultDF['Value'][etf] = result['Price']
    client.close()
    return resultDF


# def getRecordFromDateList(dateList, col_name="S&P 500"):
#     client, db, collection = setUpMongoDB(col_name=col_name)
#     resultList = []
#     for date in dateList:
#         query = {"Date": date}
#         result = collection.find_one(query)
#         if result:
#             resultList.append(result)
#     client.close()
#     return resultList
#
#
# def getRecordFromStartEnd(startDate, endDate, col_name="S&P 500"):
#     client, db, collection = setUpMongoDB(col_name=col_name)
#     resultList = []
#     date = startDate
#     while date <= endDate:
#         query = {"Date": date}
#         result = collection.find_one(query)
#         if result:
#             resultList.append(result)
#         date += datetime.timedelta(days=1)
#     return resultList


def reallocateAndGetAbsoluteReward(oldPortfolio, newPortfolio):
    '''
    oldPortfolio: {
        "portfolioDict": {"S&P 500": 0.3, "Hang Seng":0.5} -> 0.2 Cash
        "date":
        "value":
    }
    newPortfolio: {
        "portfolioDict":
        "date":
    }

    :returns: {
        oldCurrentValue: xxx,
        newCurrentValue: xxx,
        deltaValue: xxx,
        portfolio_df: portfolio_df
    }
    '''

    # 1. Check whether the input is legit
    if (
            ("portfolioDict" not in oldPortfolio) or
            ("date" not in oldPortfolio) or
            ("value" not in oldPortfolio)
    ):
        warnings.warn("Environment/calculateAbsoluteReward() Warning: Input of oldPortfolio is NOT LEGIT")
        return 0

    if (
            ("portfolioDict" not in newPortfolio) or
            ("date" not in newPortfolio)
    ):
        warnings.warn("Environment/calculateAbsoluteReward() Warning: Input of newPortfolio NOT LEGIT")
        return 0

    # 2. Check whether the portfolioDict is a dictionary
    if not isinstance(oldPortfolio['portfolioDict'], dict):
        warnings.warn(
            "Environment/calculateAbsoluteReward() Warning: oldPortfolio['portfolioDict'] is not a dictionary")
        return 0

    if not isinstance(newPortfolio['portfolioDict'], dict):
        warnings.warn(
            "Environment/calculateAbsoluteReward() Warning: newPortfolio['portfolioDict'] is not a dictionary")
        return 0

    '''
        portfolio_df:[    
            oldRatio, newRatio, oldPastValue, oldStockHeld, oldCurrentValue, oldCurrentRatio, 
            deltaRatio, deltaStockHeld, newCurrentValue
        ]
    '''
    # 3. Clean the ratio: >1: Normalize; <1: Cash Out

    oldRatio_df = pd.DataFrame.from_dict(oldPortfolio['portfolioDict'], orient='index', columns=['ratio'])
    newRatio_df = pd.DataFrame.from_dict(newPortfolio['portfolioDict'], orient='index', columns=['ratio'])
    oldRatio_df = oldRatio_df.append(pd.DataFrame(index=['CASH'], data={'ratio': np.nan}))
    newRatio_df = newRatio_df.append(pd.DataFrame(index=['CASH'], data={'ratio': np.nan}))

    if oldRatio_df['ratio'].sum() > 1:
        warnings.warn("Environment/calculateAbsoluteReward() Warning: oldRatio_df['ratio'].sum() > 1, Auto-Normalized")
        oldRatio_df = oldRatio_df / oldRatio_df['ratio'].sum()

    elif oldRatio_df['ratio'].sum() < 1:
        oldRatio_df['ratio']['CASH'] = 1 - oldRatio_df['ratio'].sum()

    if newRatio_df['ratio'].sum() > 1:
        warnings.warn(
            "Environment/calculateAbsoluteReward() Warning: newRatio_df['ratio'].values().sum() > 1, Auto-Normalized")
        newRatio_df = newRatio_df / newRatio_df['ratio'].sum()

    elif newRatio_df['ratio'].sum() < 1:
        newRatio_df['ratio']['CASH'] = 1 - newRatio_df['ratio'].sum()

    portfolio_df = pd.merge(oldRatio_df, newRatio_df, left_index=True, right_index=True, how='outer')
    portfolio_df.columns = ['oldRatio', 'newRatio']
    portfolio_df = portfolio_df.fillna(0)

    # 4. Calculate the current value of the stocks: [oldPastValue, oldStockHeld, oldCurrentValue, oldCurrentRatio]
    portfolio_df['oldPastValue'] = portfolio_df.apply(lambda row: row.oldRatio * oldPortfolio['value'], axis=1)

    etfList = list(portfolio_df.index)
    portfolio_df['oldPrice'] = getRecordFromETFList(oldPortfolio['date'], etfList)
    portfolio_df['newPrice'] = getRecordFromETFList(newPortfolio['date'], etfList)
    portfolio_df['oldStockHeld'] = portfolio_df['oldPastValue'].div(portfolio_df['oldPrice'].values)
    portfolio_df['oldCurrentValue'] = portfolio_df['oldStockHeld'].mul(portfolio_df['newPrice'].values)
    portfolio_df['oldCurrentRatio'] = portfolio_df['oldCurrentValue'] / portfolio_df['oldCurrentValue'].sum()

    # 5. Calculate the deltas [deltaRatio, deltaStockHeld, newStockHeld]

    portfolio_df['deltaRatio'] = portfolio_df['newRatio'].sub(portfolio_df['oldCurrentRatio'], fill_value=0)

    def equation(n):
        left = np.multiply(portfolio_df['oldStockHeld'] + n, portfolio_df['newPrice'])
        right = portfolio_df['newRatio'] * (
                np.dot(portfolio_df['newPrice'], portfolio_df['oldStockHeld']) - TRADING_FEE * np.dot(
            portfolio_df['newPrice'], np.absolute(n)))
        return left - right

    a0 = np.zeros(portfolio_df['oldStockHeld'].shape)
    n = fsolve(equation, a0)

    portfolio_df['deltaStockHeld'] = n
    portfolio_df['newStockHeld'] = portfolio_df['oldStockHeld'] + portfolio_df['deltaStockHeld']
    portfolio_df['newCurrentValue'] = portfolio_df['newStockHeld'].mul(portfolio_df['newPrice'])

    # 6. Return stuffs
    oldPastValueSum = portfolio_df['oldPastValue'].sum()
    newCurrentValueSum = portfolio_df['newCurrentValue'].sum()

    return {
        "oldPastValue": oldPastValueSum,
        "newCurrentValue": newCurrentValueSum,
        "deltaValue": newCurrentValueSum-oldPastValueSum,
        "portfolio_df": portfolio_df
    }
