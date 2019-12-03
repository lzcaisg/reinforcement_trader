import datetime
from Environment import *

# Test getOneDayRecord()
date = datetime.datetime(2019, 10, 10)
print(getOneDayRecord(date))


# Test getRecordFromList()
dateList = []
startDate = datetime.datetime(2019, 10, 10)
date = startDate
for i in range(30):
    dateList.append(date)
    date += datetime.timedelta(days=1)
print(getRecordFromList(dateList))


# Test getRecordFromStartEnd()
startDate = datetime.datetime(2019, 10, 10)
endDate = datetime.datetime(2019, 11, 10)

print(getRecordFromStartEnd(startDate, endDate))