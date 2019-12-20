import os
import sys
import pandas as pd
import numpy as np
import json
from MongoDBUtils import *
import datetime


def volStr2int(volStr):
    if volStr == '-':
        return 0
    elif volStr[-1] == "K":
        return int(float(volStr[:-1].replace(',', '')) * 1000)
    elif volStr[-1] == 'M':
        return int(float(volStr[:-1].replace(',', '')) * 1000000)
    elif volStr[-1] == 'B':
        return int(float(volStr[:-1].replace(',', '')) * 1000000000)
    else:
        return np.float64(volStr.replace(',', ''))


def unknown2float(numStr):
    if type(numStr) == np.float64:
        return numStr
    elif type(numStr) == float:
        return numStr
    else:
        return np.float64(numStr.replace(',', ''))


def str2date(dateStr):
    import datetime
    format_str = '%b %d, %Y'
    return datetime.datetime.strptime(dateStr, format_str)


def percent2float(percentStr):
    return float(percentStr[:-1]) / 100


def csv2db(csv_path, csv_name, etf_name):
    # ====== 1. Initial Settings ======
    # csv_path = "../index/2014-2019"
    # # file_path = ".."  # For PC
    # csv_name = "S&P 500 Historical Data.csv"
    csv_addr = os.path.join(csv_path, csv_name)

    # ====== 2. Parsing CSV to JSON ======
    csv_df = pd.DataFrame(pd.read_csv(csv_addr, sep=",", header=0, index_col=False))

    csv_df['Date'] = csv_df['Date'].apply(str2date)

    csv_df['Price'] = csv_df['Price'].apply(unknown2float)
    csv_df['Open'] = csv_df['Open'].apply(unknown2float)
    csv_df['High'] = csv_df['High'].apply(unknown2float)
    csv_df['Low'] = csv_df['Low'].apply(unknown2float)

    csv_df['Vol'] = csv_df['Vol.'].apply(volStr2int)
    csv_df.drop("Vol.", axis=1, inplace=True)  # Since MongoDB does not accept column name with dot

    csv_df['Change'] = csv_df['Change %'].apply(percent2float)
    csv_df.drop("Change %", axis=1, inplace=True)  # Since MongoDB does not accept column name with space and symbol
    # print(csv_df)
    json_str = csv_df.to_json(orient='records')
    json_list = json.loads(json_str)

    for i, v in enumerate(json_list):
        json_list[i]['Date'] = pd.to_datetime(json_list[i]['Date'], unit='ms')

    # print(json_list[0]['Date'])
    # ====== 3. Push JSON to MongoDB ======

    # client = pymongo.MongoClient(
    #     "mongodb+srv://lzcai:raspberry@freecluster-q4nkd.gcp.mongodb.net/test?retryWrites=true&w=majority")
    # # db = client['testing']
    # # collection = db['S&P500']
    # db = client[db_name]
    # collection = db[col_name]

    client, db, collection = setUpMongoDB(col_name=etf_name)

    # use collection_currency.insert(file_data) if pymongo version < 3.0
    n = len(json_list)
    for i, item in enumerate(json_list):
        collection.insert_one(item)
        sys.stdout.write("\r{0}%".format((float(i) / n) * 100))
        sys.stdout.flush()
    client.close()


csv_path = "../index/2014-2019"
# csv_path = "."  # For PC
etf_name = "Bovespa"
csv_name = etf_name+" Historical Data.csv"
csv2db(csv_path, csv_name, etf_name)