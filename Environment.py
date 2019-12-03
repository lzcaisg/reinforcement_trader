import datetime
from MongoDBUtils import *


def getOneDayRecord(date, col_name="S&P500"):
    client, db, collection = setUpMongoDB(col_name=col_name)
    myquery = {"Date": date}
    mydoc = collection.find_one(myquery)
    return mydoc

# date = datetime.datetime(2019, 10, 10)
# getOneDayRecord(date)


