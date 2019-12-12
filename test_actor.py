from Actor import Actor
import datetime

actor = Actor(todayDate=datetime.datetime(2016, 12, 10),
              startDate=datetime.datetime(2016, 9, 10),
              etfList=["FTSE 100", "Nasdaq 100", "Shanghai Composite"],
              initValue=1000000)
print(actor.etfLocalMinMaxDict['FTSE 100']['max']["180"].to_string())
print(actor.etfLocalMinMaxDict['Shanghai Composite']['min']["60"].to_string())
