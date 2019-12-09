from Environment import *
import pandas as pd

class Actor:
    MAX_WINDOW = 180
    LARGE_WINDOW = 90
    MID_WINDOW = 60
    SMALL_WINDOW = 30
    MIN_WINDOW = 7


    def __init__(self, startDate, etfList, initValue = 1000000):
        '''
        :param etflist: A list string representing ETF names, should be listed in the DB
        :param initValue: The initial amount of money, by default 1M
        '''

        # 1. Get the MAX_WINDOW day window (e.g. The first 180 day window)
        self.initialWindowDict = getRecordFromStartLengthByETFList(startDate, self.MAX_WINDOW, etfList) # Already Sorted
        '''
        initialWindowDict: {"xxx":[{Record}]}
        where Record: {"_id", "Date", "Price", "Open", "High", "Low", "Vol", "Change"}
        '''

        # 2. Get the first date
        self.etfBaseValueDict = {}
        for etf in etfList:
            self.etfBaseValueDict[etf] = self.initialWindowDict[etf][0]['Price']

        # 3. Normalize the record by the first date and store in DFs
        self.etfNormalizedDict = {}
        for etf in etfList:
            df = self.etfNormalizedDict
            baseValue = self.etfBaseValueDict[etf]
            df[etf] = pd.DataFrame(self.initialWindowDict[etf])
            df[etf]['Price'] /= baseValue
            df[etf]['Open'] /= baseValue
            df[etf]['High'] /= baseValue
            df[etf]['Low'] /= baseValue

        # 4. Find 30, 60, 90, 180 Local Minima and Maxima
        self.etfLocalMinMaxDict = {}
        for etf in etfList:
            self.etfLocalMinMaxDict[etf]={}
            df = self.etfNormalizedDict[etf]
            lastDate = np.datetime64(df['Date'].iloc[-1], 'ns')
            print(type(lastDate))
            for timeRange in [self.MAX_WINDOW, self.LARGE_WINDOW, self.MID_WINDOW, self.SMALL_WINDOW]:
                mask = (df['Date'] <= lastDate) & (df['Date'] >= lastDate-np.timedelta64(timeRange,'D'))
                self.etfLocalMinMaxDict[etf][str(timeRange)] = df.loc[mask]









