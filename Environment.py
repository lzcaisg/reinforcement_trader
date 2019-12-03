import datetime
from MongoDBUtils import *


def getOneDayRecord(date, col_name="S&P500"):
    client, db, collection = setUpMongoDB(col_name=col_name)
    query = {"Date": date}
    result = collection.find_one(query)
    client.close()
    return result


def getRecordFromList(dateList, col_name="S&P500"):
    client, db, collection = setUpMongoDB(col_name=col_name)
    resultList = []
    for date in dateList:
        query = {"Date": date}
        result = collection.find_one(query)
        if result:
            resultList.append(result)
    client.close()
    return resultList


def getRecordFromStartEnd(startDate, endDate, col_name="S&P500"):
    client, db, collection = setUpMongoDB(col_name=col_name)
    resultList = []
    date = startDate
    while date <= endDate:
        query = {"Date": date}
        result = collection.find_one(query)
        if result:
            resultList.append(result)
        date += datetime.timedelta(days=1)
    return resultList




