from Environment import *
import pandas as pd


class Actor:
    MAX_WINDOW = 180
    LARGE_WINDOW = 90
    MID_WINDOW = 60
    SMALL_WINDOW = 30
    MIN_WINDOW = 7
    TOP_PERCENTILE = 10
    initialWindowDict = {}
    etfBaseValueDict = {}
    etfNormalizedDict = {}
    etfLocalMinMaxDict = {}


    def __init__(self, startDate, etfList, initValue=1000000):
        '''
        :param etflist: A list string representing ETF names, should be listed in the DB
        :param initValue: The initial amount of money, by default 1M
        '''

        # 1. Get the MAX_WINDOW day window (e.g. The first 180 day window)
        self.initialWindowDict = getRecordFromStartLengthByETFList(startDate, self.MAX_WINDOW,
                                                                   etfList)  # Already Sorted
        '''
        initialWindowDict: {"S&P 500":[{__Record__}, {__Record__}, ...]}
        where Record: {"_id", "Date", "Price", "Open", "High", "Low", "Vol", "Change"}
        '''

        # 2. Get the first date
        for etf in etfList:
            self.etfBaseValueDict[etf] = self.initialWindowDict[etf][0]['Price']

        # 3. Normalize the record by the first date and store in DFs
        for etf in etfList:
            baseValue = self.etfBaseValueDict[etf]
            self.etfNormalizedDict[etf] = pd.DataFrame(self.initialWindowDict[etf])

            df = self.etfNormalizedDict[etf]
            df['Price'] /= baseValue
            df['Open'] /= baseValue
            df['High'] /= baseValue
            df['Low'] /= baseValue

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

            df = self.etfNormalizedDict[etf]
            lastDate = np.datetime64(df['Date'].iloc[-1], 'ns')
            for timeRange in [self.MAX_WINDOW, self.LARGE_WINDOW, self.MID_WINDOW, self.SMALL_WINDOW]:
                mask = (df['Date'] <= lastDate) & (df['Date'] >= lastDate - np.timedelta64(timeRange, 'D'))
                # lenth of the nMax DF
                nSample = int(df.loc[mask].shape[0] * self.TOP_PERCENTILE / 100) + 1
                self.etfLocalMinMaxDict[etf]['max'][str(timeRange)] = df.loc[mask].nlargest(nSample, 'Price')
                self.etfLocalMinMaxDict[etf]['min'][str(timeRange)] = df.loc[mask].nsmallest(nSample, 'Price')
