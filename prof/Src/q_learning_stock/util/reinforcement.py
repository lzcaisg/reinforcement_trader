import numpy as np
import pandas as pd
import math

# prints formatted price
def format_price(price) -> str:
	return ("-$" if price < 0 else "$") + "{0:.2f}".format(abs(price))

# returns all close prices from file as a vector
def get_close_array(key: str) -> np.array:
	df = pd.read_csv("data\\{}.csv".format(key), usecols=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'], parse_dates=['Date'])
	return df["Close"].to_numpy()

# returns the sigmoid
def sigmoid(x):
	return 1 / (1 + math.exp(-x))

# returns an an n-day state representation ending at time t
def get_state(data: np.array, t, n) -> np.array:
	d = t - n + 1
	print("t: {}, n: {}, d: {}".format(t,n,d))
	block = data[d:t+1] if d >= 0 else -d * [data[0]] + data[0:t+1] # pad with t0
	print(block)
	print(len(block))
	res = []
	for i in range(n - 1):
		res.append(sigmoid(block[i + 1] - block[i]))

	return np.array([res])