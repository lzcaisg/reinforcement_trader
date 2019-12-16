from Environment import *
import pandas as pd

MAX_WINDOW = 180
LARGE_WINDOW = 90
MID_WINDOW = 60
SMALL_WINDOW = 30
MIN_WINDOW = 7

class Actor:
    TOP_PERCENTILE = 10
    initialWindow_dictList = {}
    etfBaseValueDict = {}
    etfNormalized_dfDict = {}
    etfLocalMinMaxDict = {}
    todayDate = None


    def __init__(self, todayDate, startDate, etfList, initValue=1000000):
        '''
        :param todayDate: If todayDate <= startDate + 180d, go back from todayDate; otherwase go forward from startDate
        :param etflist: A list string representing ETF names, should be listed in the DB
        :param initValue: The initial amount of money, by default 1M
        '''

        # In any cases, we shouldn't know today's value:
        # Therefore, if there is any record goes beyond today, we should delete it

        # 1. Get the MAX_WINDOW day window (e.g. The first 180 day window)
        self.env = Environment()
        self.todayDate = todayDate
        self.initialWindow_dictList = \
            self.env.getRecordFromStartLengthByETFList(
                todayDate, startDate, MAX_WINDOW, etfList)  # Already Sorted
        '''
        initialWindowDict: {"S&P 500":[{__Record__}, {__Record__}, ...]}
        where Record: {"_id", "Date", "Price", "Open", "High", "Low", "Vol", "Change"}
        '''

        # 2. Get the first date
        try:
            for etf in etfList:
                self.etfBaseValueDict[etf] = self.initialWindow_dictList[etf][0]['Price']
        except Exception as e: # If there is no value returned; e,g, todayDate < StartDate
            self.etfBaseValueDict[etf] = {}
            print("Error in Actor.etfBaseValueDict: When visiting self.initialWindowDict[etf][0]['Price'],", e)

        # 3. Normalize the record by the first date and store in DFs

        for etf in etfList:
            try:
                baseValue = self.etfBaseValueDict[etf]
                self.etfNormalized_dfDict[etf] = pd.DataFrame(self.initialWindow_dictList[etf])

                df = self.etfNormalized_dfDict[etf]
                df['Price'] /= baseValue
                df['Open'] /= baseValue
                df['High'] /= baseValue
                df['Low'] /= baseValue
            except Exception as e:  # If there is no value returned; e,g, todayDate < StartDate
                df['Price'] = np.nan
                df['Open'] = np.nan
                df['High'] = np.nan
                df['Low'] = np.nan
                print("Error in Actor.etfNormalizedDict: When visiting self.etfBaseValueDict[etf],", e)

        # 4. Find 30, 60, 90, 180 Local Minima and Maxima (Top a% Percentile)
        '''
        self.etfLocalMinMaxDict["FTSE 100"]['max']['180'] = 
                                 _id       Date     Price      Open      High       Low        Vol  Change
        31  5de87f153b02dbc0c223fea8 2015-10-23  1.046829  1.035815  1.053946  1.035815  749540000  0.0106
        34  5de87f153b02dbc0c223fea5 2015-10-28  1.045809  1.034026  1.047540  1.032485  819720000  0.0114
        59  5de87f143b02dbc0c223fe8c 2015-12-02  1.043068  1.038962  1.047359  1.038888  583610000  0.0040
        32  5de87f153b02dbc0c223fea7 2015-10-26  1.042433  1.046829  1.048278  1.040542  454680000 -0.0042
        21  5de87f153b02dbc0c223feb2 2015-10-09  1.042293  1.035578  1.048314  1.035578  768200000  0.0065        
        '''
        for etf in etfList:
            self.etfLocalMinMaxDict[etf] = {}
            self.etfLocalMinMaxDict[etf]['max'] = {}
            self.etfLocalMinMaxDict[etf]['min'] = {}
            try:
                df = self.etfNormalized_dfDict[etf]
                lastDate = np.datetime64(df['Date'].iloc[-1], 'ns')
                for timeRange in [MAX_WINDOW, LARGE_WINDOW, MID_WINDOW, SMALL_WINDOW]:
                    mask = (df['Date'] <= lastDate) & (df['Date'] >= lastDate - np.timedelta64(timeRange, 'D'))
                    nSample = int(df.loc[mask].shape[0] * self.TOP_PERCENTILE / 100) + 1  #nSample: lenth of the minmax DF

                    self.etfLocalMinMaxDict[etf]['max'][str(timeRange)] = df.loc[mask].nlargest(nSample, 'Price')
                    self.etfLocalMinMaxDict[etf]['min'][str(timeRange)] = df.loc[mask].nsmallest(nSample, 'Price')

            except Exception as e:  # If there is no value returned; e,g, todayDate < StartDate
                for timeRange in [MAX_WINDOW, LARGE_WINDOW, MID_WINDOW, SMALL_WINDOW]:
                    self.etfLocalMinMaxDict[etf]['max'][str(timeRange)] = None
                    self.etfLocalMinMaxDict[etf]['min'][str(timeRange)] = None
                print("Error in Actor.etfLocalMinMaxDict: When visiting self.etfNormalizedDict[etf],", e)

    def predrict(self, endDate): # Start Predicting from todayDate to endDate
        pass