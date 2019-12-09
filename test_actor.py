from Actor import Actor
import datetime

actor = Actor(datetime.datetime(2015, 9, 10), ["FTSE 100", "Nasdaq 100", "Shanghai Composite"], initValue = 1000000)
print(actor.etfLocalMinMaxDict['FTSE 100'])

