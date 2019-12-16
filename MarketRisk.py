from Environment import *
import numpy as np
from matplotlib import pyplot as plt

MAX_WINDOW = 180
LARGE_WINDOW = 90
MID_WINDOW = 60
SMALL_WINDOW = 30
MIN_WINDOW = 7

env = Environment()
currentDate = EARLIEST_DATE+datetime.timedelta(days=SMALL_WINDOW-1) # SMALL_WINDOW: 30 days
etfName = "S&P 500"
marketName = "MSCI World"

dbResult = env.getRecordFromEndLengthByETFList(
    todayDate=datetime.datetime.now(),
    endDate=currentDate,
    length=SMALL_WINDOW-1,
    etfList=[etfName, marketName])
if len(dbResult[etfName]) != len(dbResult[marketName]):
    delta = len(dbResult[etfName]) - len(dbResult[marketName])
    if delta > 0: # etf got more data:
        for i in range(delta):
            dbResult[marketName].insert(0, dbResult[marketName][0])
    else: # market got more data:
        for i in range(abs(delta)):
            dbResult[etfName].insert(0, dbResult[etfName][0])

currentDate += datetime.timedelta(days=1)
betaDict = {}

counter = 0

while currentDate < LATEST_DATE:
    # 1. Push the record of currentDate as the first item of dbResult
    newResult = env.getRecordFromStartLengthByETFList(datetime.datetime.now(), currentDate, 1, [etfName, marketName])
    if newResult[etfName]: # If there IS a new record
        dbResult[etfName].insert(0, newResult[etfName][0])
        if newResult[marketName]: # The dimension must match
            dbResult[marketName].insert(0, newResult[marketName][0])
        else:
            dbResult[marketName].insert(0, dbResult[marketName][0])


    marketChange = np.array([d['Change'] for d in dbResult[marketName]])
    etfChange = np.array([d['Change'] for d in dbResult[etfName]])
    beta = (np.cov(etfChange,marketChange)[0][1])/np.var(marketChange)
    betaDict[currentDate] = beta
    currentDate += datetime.timedelta(days=1)

    if counter%200 == 0:
        print(counter, beta)
    counter += 1

print(betaDict.items())
lists = sorted(betaDict.items()) # sorted by key, return a list of tuples

x, y = zip(*lists) # unpack a list of pairs into two tuples

plt.plot(x, y)
plt.show()

