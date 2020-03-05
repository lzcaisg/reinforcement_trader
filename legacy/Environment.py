import datetime
import warnings
import pandas as pd
import numpy as np
from MongoDBUtils import *
from scipy.optimize import fsolve
import pymongo

TRADING_FEE = 0.008
EARLIEST_DATE = datetime.datetime(2014, 10, 17)
LATEST_DATE = datetime.datetime(2019, 10, 17)


# In any cases, we shouldn't know today's and future value;
# ONLY PROVIDE CALCULATED RESULT
# Handled by Both Environment and Actors

class Environment():
    def __init__(self):
        self.client = pymongo.MongoClient("mongodb+srv://lzcai:raspberry@freecluster-q4nkd.gcp.mongodb.net/test?retryWrites=true&w=majority")
        self.db = self.client["testing"]



    def getOneRecord(self, todayDate, date, col_name="S&P 500"):
        '''

        :param todayDate:
        :param date:
        :param col_name:
        :return: e.g.
            {
                '_id': ObjectId('5de7325e05597fc4f7b09fad'),
                'Date': datetime.datetime(2019, 9, 10, 0, 0),
                'Price': 2979.39, 'Open': 2971.01,
                'High': 2979.39,
                'Low': 2957.01,
                'Vol': 0,
                'Change': 0.0003
            }

        '''
        if date >= todayDate:
            return
        collection = self.db[col_name]
        query = {"Date": date}
        result = collection.find_one(query)
        return result


    def getAllRecord(self, todayDate, col_name="S&P 500"):
        pass



    def getRecordFromDateList(self, todayDate, dateList, col_name="S&P 500"):
        collection = self.db[col_name]
        resultList = []
        for date in dateList:
            if date >= todayDate:
                continue
            query = {"Date": date}
            result = collection.find_one(query)
            if result:
                resultList.append(result)
        return resultList


    def getRecordFromStartLength(self, todayDate, startDate, length, col_name="S&P 500"): # Return Sorted List of Dict
        collection = self.db[col_name]
        resultList = []
        for i in range(length):
            newDate = startDate + datetime.timedelta(days=i)
            if newDate >= todayDate:
                break
            query = {"Date": newDate}
            result = collection.find_one(query)
            if result:
                resultList.append(result)
        return resultList


    def getRecordFromStartLengthByETFList(self, todayDate, startDate, length, etfList):
        '''

        :param startDate:
        :param length:
        :param etfList: ["S&P 500", "DAX"]
        :return: A Dict
            {
            "S&P 500": [{one record}, {another record}],
            "DAX":[{...}, {...}],
            ...}
        '''
        if not isinstance(etfList, list):
            warnings.warn("Environment/getRecordFromStartLengthByETFList() Warning: etfList is not List")
            return None
        resultDict = {}

        for etf in etfList:
            if etf == "CASH":
                continue
            else:
                etfRecordList = []
                collection = self.db[etf]
                for i in range(length):
                    newDate = startDate + datetime.timedelta(days=i)
                    if newDate >= todayDate:
                        break

                    query = {"Date": newDate}
                    result = collection.find_one(query)
                    if result:
                        etfRecordList.append(result)
                resultDict[etf] = etfRecordList

        return resultDict


    def getRecordFromEndLengthByETFList(self, todayDate, endDate, length, etfList):
        '''

        :param startDate:
        :param length:
        :param etfList: ["S&P 500", "DAX"]
        :return: A Dict
            {
            "S&P 500": [{one record}, {another record}],
            "DAX":[{...}, {...}],
            ...}
        '''
        if not isinstance(etfList, list):
            warnings.warn("Environment/getRecordFromStartLengthByETFList() Warning: etfList is not List")
            return None

        resultDict = {}

        for etf in etfList:
            if etf == "CASH":
                continue
            else:
                etfRecordList = []
                collection = self.db[etf]
                for i in range(length):
                    newDate = endDate - datetime.timedelta(days=i)
                    if newDate >= todayDate:
                        continue

                    query = {"Date": newDate}
                    result = collection.find_one(query)
                    if result:
                        etfRecordList.append(result)
                resultDict[etf] = etfRecordList

        return resultDict


    def getPriceByETFList(self, todayDate, date, etfList):  # Get PRICE only! Not the full record
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

        resultDF = pd.DataFrame(etfList, columns=["Name"]).set_index('Name', drop=True)
        resultDF['Value'] = np.nan

        for etf in etfList:
            if etf == "CASH":
                resultDF['Value'][etf] = 1
            else:
                collection = self. db[etf]
                if date >= todayDate:
                    continue
                query = {"Date": date}
                result = collection.find_one(query)
                if result:
                    resultDF['Value'][etf] = result['Price']
        return resultDF


    def reallocateAndGetAbsoluteReward(self, oldPortfolio, newPortfolio):
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
            warnings.warn(
                "Environment/calculateAbsoluteReward() Warning: oldRatio_df['ratio'].sum() > 1, Auto-Normalized")
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
        portfolio_df['oldPrice'] = self.getPriceByETFList(oldPortfolio['date'], etfList)
        portfolio_df['newPrice'] = self.getPriceByETFList(newPortfolio['date'], etfList)
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
            "deltaValue": newCurrentValueSum - oldPastValueSum,
            "portfolio_df": portfolio_df
        }


    def getFuturePercentile(self, todayDate, delta, col_name="S&P 500"): # Delta includes todayDate!
        # 1. To get all future results ang calculate the percentile using getRecordFromStartLength
        # Disable the today_check by passing real-world date
        resultList = self.getRecordFromStartLength(datetime.datetime.now(), todayDate, delta, col_name=col_name)

        # 2. Transform the resultList into dataframe

        df = pd.DataFrame(resultList)
        todayRank = df['Price'].rank(method = 'average')[0]  # The smaller the value, the smaller the rank
        todayPercentile = (todayRank-1) / (df.shape[0]-1) # -1 to make it [0, 1], otherwise rank start with 1
        # The greater the percentile, the worse the performance in the future
        return todayPercentile
