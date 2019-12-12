from Actor import Actor
import datetime

actor = Actor(datetime.datetime(2015, 9, 10), ["FTSE 100", "Nasdaq 100", "Shanghai Composite"], initValue=1000000)
print(actor.etfLocalMinMaxDict['FTSE 100']['max']["180"].to_string())
print(actor.etfLocalMinMaxDict['FTSE 100']['min']["60"].to_string())
