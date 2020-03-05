import datetime
from Environment import *
import torch

# Test getOneDayRecord()
# print(getOneRecord(datetime.datetime(2020, 10, 10), datetime.datetime(2019, 9, 10)))

# # Test getRecordFromList()
# # dateList = []
# # startDate = datetime.datetime(2019, 10, 10)
# # date = startDate
# # for i in range(30):
# #     dateList.append(date)
# #     date += datetime.timedelta(days=1)
# # print(getRecordFromDateList(dateList))
# #
# #
# # # Test getRecordFromStartEnd()
# # startDate = datetime.datetime(2019, 10, 10)
# # endDate = datetime.datetime(2019, 11, 10)
# #
# # print(getRecordFromStartEnd(startDate, endDate))


# oldPortfolio = {
#     "portfolioDict": {"S&P 500": 0.3, "Taiwan Weighted": 0.5},
#     "date": datetime.datetime(2019, 10, 7),
#     "value": 1000000
# }
#
# newPortfolio = {
#     "portfolioDict": {"S&P 500": 0.5, "Taiwan Weighted": 0.4},
#     "date": datetime.datetime(2019, 10, 17),
# }
#
# print(reallocateAndGetAbsoluteReward(oldPortfolio, newPortfolio)['portfolio_df'].to_string())


# Test getRecordFromStartEnd()
# startDate = datetime.datetime(2019, 9, 10)
#
# print(getRecordFromStartLengthByETFList(startDate, 3, ["S&P 500", "Taiwan Weighted"]))


# Test getFuturePercentile()
startDate = datetime.datetime(2020, 9, 10)
env = Environment()
# print(env.getFuturePercentile(startDate, 180))

print(env.getRecordFromStartLengthByETFList(datetime.datetime.now(),startDate,1,["S&P 500"]))