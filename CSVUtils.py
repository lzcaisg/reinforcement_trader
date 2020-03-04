import numpy as np
import os
import pandas as pd

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


def csv2df(csv_path, csv_name, source = "investing"):
    # ====== 1. Initial Settings ======
    # csv_path = "../index/2014-2019"
    # # file_path = ".."  # For PC
    # csv_name = "S&P 500 Historical Data.csv"
    csv_addr = os.path.join(csv_path, csv_name)

    # ====== 2. Parsing CSV to JSON ======
    csv_df = pd.DataFrame(pd.read_csv(csv_addr, sep=",", header=0, index_col=False))

    if source == "investing":
        # csv_df['Date'] = csv_df['Date'].apply(str2date)
        csv_df['Date'] = pd.to_datetime(csv_df['Date'])
        csv_df['Price'] = csv_df['Price'].apply(unknown2float)
        csv_df['Open'] = csv_df['Open'].apply(unknown2float)
        csv_df['High'] = csv_df['High'].apply(unknown2float)
        csv_df['Low'] = csv_df['Low'].apply(unknown2float)

        # print(csv_df['Low'])
        # if "Vol." in list(csv_df.columns):
	       #  csv_df['Vol'] = csv_df['Vol.'].apply(volStr2int)
	       #  csv_df.drop("Vol.", axis=1, inplace=True)  # Since MongoDB does not accept column name with dot

        csv_df['Change'] = csv_df['Change %'].apply(percent2float)
        csv_df.drop("Change %", axis=1, inplace=True)  # Since MongoDB does not accept column name with space and symbol
        # print(csv_df)
    
    elif source == "yahoo":
        csv_df.drop("Close", axis=1, inplace=True)
        csv_df.columns = ["Date", "Open", "High", "Low", "Price", "Vol"]
        csv_df['Date'] = pd.to_datetime(csv_df['Date'])
        csv_df['Change'] = csv_df['Price'].pct_change()
    
    return csv_df

