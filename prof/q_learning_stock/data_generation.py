#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'eddy_src/q_learning_stock'))
	print(os.getcwd())
except:
	pass

import indicators
import util
import config
import numpy as np
import pandas as pd
import scipy
import time
import calendar
import math

#%% [markdown]
## Data Generation script for csv data. Use this script to generate data for missing csv data. For scripts using Alpha Vantage, a user key should be obtained

#%% [markdown]
### Generate fundamental data for JP morgan: JPM

#%%
# df = pd.read_csv('data/fundamental.csv')
# df = df.drop(['SimFin ID', 'Company Industry Classification Code'], axis=1)
# df = df.pivot_table(index=['Ticker','publish date', 'Indicator Name'],
# 					values=['Indicator Value'])
# jpm = df.iloc[df.index.get_level_values('Ticker')=='JPM']
# jpm.to_csv('data/jpm_fundamental.csv')

#%% [markdown]
### Generate dataset for portfolios
#%%
def generate_portfolio_dataset(stocks: list, path: str, start='1/1/2016', end='31/12/2018'):
	"""Note that api might not respond fast enough, or have a response limit.
	"""
	for stock in stocks:
		try:
			df = util.get_stock_data(stock, start, end)
			df.index.name = 'Date'
			filename_path = os.path.join(path, stock)
			df.to_csv('{}.csv'.format(filename_path), header=['Open', 'High', 'Low', 'Close', 'Volume'])
			print("{} done".format(stock))
			time.sleep(1)
		except Exception as e:
			print("Unable to fetch data for {}".format(stock))
			print(e)

#%% [markdown]
#### Generate stock data for various market indexes
##### CAC 40: ^FCHI
##### DAX PERFORMANCE-INDEX: ^GDAXI
##### Dow Jones Industrial Average: ^DJI
##### S&P 500 Index: ^GSPC
##### S&P 100: ^OEX
##### FTSE 100 Index: ^FTSE
##### Nikkei 225: ^N225
##### SSE Composite Index: ^SSEC
##### NYSE Composite: ^NYA
##### Euro Stoxx 50: ^STOXX50E
##### Technology Select Sector SPDR ETF: XLK
##### Energy Select Sector SPDR Fund: XLE
##### Health Care Select Sector SPDR Fund: XLV
##### Consumer Discret Sel Sect SPDR ETF: XLY

#%%
stocks = ['^FCHI', '^GDAXI', '^DJI', '^GSPC', '^OEX', '^FTSE', '^N225', '^SSEC', '^NYA', '^STOXX50E','XLK', 'XLE', 'XLV', 'XLY']
generate_portfolio_dataset(stocks, 'data/indexes')

#%% [markdown]
## Generate for algo
### IBOVESPA (Brazil): ^BVSP
### TSEC weighted index: ^TWII
### NASDAQ Composite: ^IXIC
#%%
# alpha vantage failed to fetch for nasdaq composite. Data can be downloaded here: https://www.quandl.com/data/NASDAQOMX/COMP-NASDAQ-Composite-COMP manually
stocks = ['^BVSP', '^TWII']
generate_portfolio_dataset(stocks, 'data/indexes', '1/1/2014')

#%% [markdown]
##Generate some stocks for testing in algo
### BVSP
#### Equatorial Energia S.A.: EQTL3.SA
#### Itaúsa - Investimentos Itaú S.A.: ITSA4.SA
#### Petróleo Brasileiro S.A. - Petrobras: PETR3.SA
#%%
stocks = ['EQTL3.SA', 'ITSA4.SA', 'PETR3.SA']
generate_portfolio_dataset(stocks, 'data/algo/^BVSP', '1/1/2014')

#%% [markdown]
### TWII
#### Formosa Chemicals & Fibre Corporation: 1326.TW
#### LARGAN Precision Co.,Ltd: 3008.TW
#### Cathay Financial Holding Co., Ltd. 2882.TW
#%%
stocks = ['1326.TW', '3008.TW', '2882.TW']
generate_portfolio_dataset(stocks, 'data/algo/^TWII', '1/1/2014')

#%% [markdown]
### IXIC
#### Tesla, Inc.: TSLA
#### IBERIABANK Corporation: IBKC
#### FireEye, Inc.: FEYE
#%%
stocks = ['TSLA', 'IBKC', 'FEYE']
generate_portfolio_dataset(stocks, 'data/algo/^IXIC', '1/1/2014')

#%% [markdown]
#### Generate stock data for Goldman Sachs
#### GS Equity Growth Portfolio - Institutional
##### International Equity Insights Fund Institutional Class: GCIIX
##### Goldman Sachs Large Cap Growth Insights Fund Institutional Class: GCGIX
##### Goldman Sachs Large Cap Value Insights Fund Institutional Class: GCVIX
##### Goldman Sachs Emerging Markets Equity Insights Fund International: GERIX
##### Goldman Sachs ActiveBeta U.S. Large Cap Equity ETF: GSLC
##### Goldman Sachs ActiveBeta Emerging Markets Equity ETF: GEM
##### GS Global Real Estate Securities Fund: GARSX
##### Goldman Sachs Small Cap Equity Insights Fund Institutional Class: GCSIX
##### Goldman Sachs ActiveBeta International Equity ETF: GSIE
##### GS Financial Square Government Fund: FGTXX
##### Goldman Sachs International Small Cap Insights Fund Institutional Class: GICIX
##### iShares MSCI Brazil Capped ETF: EWZ

#%%
# Note that 'FGTXX' is ignored due to having no data avaliable. It consists of only U.S. Government and U.S. Treasury securities including bills, bonds, notes and repurchase agreements, which are stable
stocks = ['GCIIX', 'GCGIX', 'GCVIX', 'GERIX', 'GSLC', 'GEM', 'GARSX', 'GCSIX', 'GSIE', 'GICIX', 'EWZ']
generate_portfolio_dataset(stocks, 'data/goldman')


#%% [markdown]
#### Generate quarterly returns for GS Portfolios
##### Obtain stock data for Balanced Strategy Portfolio: GIPAX, GIPCX, GIPIX, GIPSX, GIPTX, GIPRX, GIPUX

#%%
stocks = ['GIPAX', 'GIPCX', 'GIPIX', 'GIPSX', 'GIPTX', 'GIPRX', 'GIPUX']
generate_portfolio_dataset(stocks, 'data/goldman/portfolio/balanced')

#%% [markdown]
##### Obtain stock data for Equity Growth Strategy Portfolio: GAPAX, GAXCX, GAPIX, GAPSX, GAPTX, GAPRX, GAPUX

#%%
stocks = ['GAPAX', 'GAXCX', 'GAPIX', 'GAPSX', 'GAPTX', 'GAPRX', 'GAPUX']
generate_portfolio_dataset(stocks, 'data/goldman/portfolio/equity_growth')

#%% [markdown]
##### Obtain stock data for Growth and Income Strategy Portfolio: GOIAX, GOICX, GOIIX, GOISX, GPITX, GPIRX, GOIUX

#%%
stocks = ['GOIAX', 'GOICX', 'GOIIX', 'GOISX', 'GPITX', 'GPIRX', 'GOIUX']
generate_portfolio_dataset(stocks, 'data/goldman/portfolio/growth_income')

#%% [markdown]
##### Obtain stock data for Growth Strategy Portfolio: GGSAX, GGSCX, GGSIX, GGSSX, GGSTX, GGSRX, GGSUX

#%%
stocks = ['GGSAX', 'GGSCX', 'GGSIX', 'GGSSX', 'GGSTX', 'GGSRX', 'GGSUX']
generate_portfolio_dataset(stocks, 'data/goldman/portfolio/growth')

#%% [markdown]
##### Obtain stock data for Satellite Strategy Portfolio: GXSAX, GXSCX, GXSIX, GXSSX, GXSTX, GXSRX, GXSUX

#%%
stocks = ['GXSAX', 'GXSCX', 'GXSIX', 'GXSSX', 'GXSTX', 'GXSRX', 'GXSUX']
generate_portfolio_dataset(stocks, 'data/goldman/portfolio/satellite')

#%% [markdown]
#### Generate quarterly returns for portfolios
#%%
def generate_portfolio_quarterly_returns(directory_path: str):
	"""Generate portfolio_quarterly_return.csv in path. Not recursive.
	"""
	quarterly_dict = {'symbol': [], 'start_period': [], 'end_period': [], 'quarterly_return': []}
	print('Found the following files in directory: {}'.format(os.listdir(directory_path)))
	for filename in os.listdir(directory_path):
		not_stocks = ['portfolio_quarterly_return.csv', 'pearson_correlation.csv', 
		'portfolio_quarterly_return.csv', 'best_portfolio_switch.csv']
		if filename.endswith(".csv") and filename not in not_stocks:
			print('Processing {}...'.format(filename))
			file_path = os.path.join(directory_path, filename)
			df = pd.read_csv(file_path, parse_dates=['Date'])
			start_year = df['Date'].iloc[0].year
			end_year = df['Date'].iloc[-1].year
			symbol = filename[:-4]
			for year in range(start_year, end_year + 1):
				for quarter_start in range(1,13,3):
					q_start = '{}-{}-1'.format(year, quarter_start)
					q_end = '{}-{}-{}'.format(year, quarter_start + 2, calendar.monthrange(year, quarter_start + 2)[1])
					temp_df = df[(df['Date'] >= q_start) & (df['Date'] <= q_end)]
					# Remove empty data
					temp_df = temp_df[temp_df['Close'] != 0]
					if len(temp_df.index) > 2:
						quarter_start_close = temp_df['Close'].iloc[0]
						quarter_end_close = temp_df['Close'].iloc[-1]
						quarter_return = (quarter_end_close - quarter_start_close) / quarter_start_close * 100
						quarterly_dict['symbol'].append(symbol)
						quarterly_dict['start_period'].append(temp_df['Date'].iloc[0].date())
						quarterly_dict['end_period'].append(temp_df['Date'].iloc[-1].date())
						quarterly_dict['quarterly_return'].append(quarter_return)
	df = pd.DataFrame(quarterly_dict)
	report_path = os.path.join(directory_path, 'portfolio_quarterly_return.csv')
	df.to_csv(report_path)

#%%
##### Generate quarterly returns for all portfolios in goldman
portfolios = ['balanced', 'equity_growth', 'growth', 'growth_income', 'satellite']
for portfolio in portfolios:
	try:
		portfolio_path = os.path.join('data/goldman/portfolio', portfolio)
		generate_portfolio_quarterly_returns(portfolio_path)
		print('Generated report for {}'.format(portfolio))
	except Exception as e:
		print('Error generating report for {}'.format(portfolio))
		print(e)
print('Report generation done.')

#%% [markdown]
##### Generate best portfolio switch for goldman
###### Concatenate all portfolio returns
#%%
portfolios = ['balanced', 'equity_growth', 'growth', 'growth_income', 'satellite']
portfolio_dfs = []
for portfolio in portfolios:
	portfolio_report_path = os.path.join('data/goldman/portfolio', portfolio, 'portfolio_quarterly_return.csv')
	portfolio_dfs.append(pd.read_csv(portfolio_report_path))
summary_df = pd.concat(portfolio_dfs).reset_index()
summary_df.to_csv('data/goldman/summary.csv', index=False)

#%% [markdown]
###### Obtain max returns per period

#%%
best_portfolio_switch_df = summary_df.loc[summary_df.groupby(['start_period', 'end_period'])['quarterly_return'].idxmax()]
best_portfolio_switch_df.to_csv('data/goldman/best_portfolio_switch.csv')

#%% [markdown]
##### Generate Portfolio quartery returns for indexes

#%%
portfolio_path = os.path.join('data', 'indexes')
generate_portfolio_quarterly_returns(portfolio_path)

#%% [markdown]
###### Obtain max returns per period

#%%
index_df = pd.read_csv('data/indexes/portfolio_quarterly_return.csv')
best_portfolio_switch_df = index_df.loc[index_df.groupby(['start_period', 'end_period'])['quarterly_return'].idxmax()]
best_portfolio_switch_df = best_portfolio_switch_df.drop([best_portfolio_switch_df.columns[0]], axis=1)
best_portfolio_switch_df.to_csv('data/indexes/best_portfolio_switch.csv')

#%% [markdown]
## Create pearson correlation for stocks in folder

#% [markdown]
### Pearson correlation for indexes using DJI as base
#%%
base = '^DJI'
stocks = []
for filename in os.listdir('data/indexes'):
	not_stocks = ['portfolio_quarterly_return.csv', 'pearson_correlation.csv', 
	'portfolio_quarterly_return.csv', 'best_portfolio_switch.csv']
	if filename.endswith(".csv") and filename not in not_stocks:
		symbol = filename[:-4]
		if symbol != base:
			stocks.append(symbol)
print(stocks)
symbol = []
corr_arr = []
p_value_arr = []
base_df = pd.read_csv('data/indexes/{}.csv'.format(base))
# Remove empty data
base_df = base_df[base_df['Close'] != 0]

for stock in stocks:
	print('Processing {}...'.format(stock))
	temp_df = pd.read_csv('data/indexes/{}.csv'.format(stock))
	# Remove empty data
	temp_df = temp_df[temp_df['Close'] != 0]
	# Only include data they both have
	temp_adjusted_base_df = base_df[base_df['Date'].isin(temp_df['Date'])]
	temp_df = temp_df[temp_df['Date'].isin(temp_adjusted_base_df['Date'])]
	print('Stock: {}, base len: {}, stock len: {}'.format(stock, len(temp_adjusted_base_df), len(temp_df)))
	# Obtain correlation
	corr, p_value = scipy.stats.pearsonr(temp_adjusted_base_df['Close'], temp_df['Close'])
	symbol.append(stock)
	corr_arr.append(corr)
	p_value_arr.append(p_value)
pearson_df = pd.DataFrame({'Symbol': symbol, 'Correlation': corr_arr, 'p-value': p_value_arr})
pearson_df.to_csv('data/indexes/pearson_correlation.csv')
print('Saved.')

#%% [markdown]
### Calculate gradient of EMA in index
#%%
stocks = []
for filename in os.listdir('data/indexes'):
	not_stocks = ['portfolio_quarterly_return.csv', 'pearson_correlation.csv', 
	'portfolio_quarterly_return.csv', 'best_portfolio_switch.csv']
	if filename.endswith(".csv") and filename not in not_stocks:
		symbol = filename[:-4]
		stocks.append(symbol)
summary_arr = []
for stock in stocks:
	print('Processing {}...'.format(stock))
	df = pd.read_csv('data/indexes/{}.csv'.format(stock), parse_dates=['Date'])
	df['EMA'] = indicators.exponential_moving_avg(df, center=False)
	df['Symbol'] = [stock] * len(df)
	from sklearn.preprocessing import MinMaxScaler
	scaler = MinMaxScaler()
	# ema_1d_gradient
	ema_1d_gradient = [float('nan')]
	for i in range(1,len(df)):
		arr = []
		for j in range(1, -1, -1):
			arr.append(df['EMA'].iloc[i-j])
		ema_1d_gradient.append(np.gradient(np.array(arr)).mean())
	df['ema_1d_gradient'] = scaler.fit_transform(np.array(ema_1d_gradient).reshape(-1, 1))
	# ema_3d_gradient
	ema_3d_gradient = [float('nan')] * 3
	for i in range(3,len(df)):
		arr = []
		for j in range(3, -1, -1):
			arr.append(df['EMA'].iloc[i-j])
		ema_3d_gradient.append(np.gradient(np.array(arr)).mean())
	df['ema_3d_gradient'] = scaler.fit_transform(np.array(ema_3d_gradient).reshape(-1, 1))
	# ema_5d_gradient
	ema_5d_gradient = [float('nan')] * 5
	for i in range(5,len(df)):
		arr = []
		for j in range(5, -1, -1):
			arr.append(df['EMA'].iloc[i-j])
		ema_5d_gradient.append(np.gradient(np.array(arr)).mean())
	df['ema_5d_gradient'] = scaler.fit_transform(np.array(ema_5d_gradient).reshape(-1, 1))
	df['MACD Line'] = indicators.macd_line(df, center=False)
	df['MACD Signal'] = indicators.macd_signal(df, center=False)
	df['+MACD Line'] = df['MACD Line'][df['MACD Line'] > df['MACD Signal']]
	df['-MACD Line'] = df['MACD Line'][df['MACD Line'] < df['MACD Signal']]
	df['buy_gradient'] = np.gradient(np.array(df['EMA'][df['MACD Line'] > df['MACD Signal']]))
	summary_arr.append(df)
summary_df = pd.concat(summary_arr).reset_index()
summary_df.to_csv('data/indexes/metadata/gradient_summary.csv', index=False)
print('Saved')

#%%
df = pd.read_csv('data/indexes/metadata/gradient_summary.csv').reset_index()
buy_period_dict = {'Symbol': [], 'Buy_period': []}
for i, row in enumerate(df.groupby(df['Symbol'])['+MACD Line']):
	# Note: row is a tuple with (symbol, values)
	current_period = []
	for j, value in enumerate(row[1]):
		# Group consecutive periods without nan together
		if math.isnan(value):
			if len(current_period) != 0:
				buy_period_dict['Symbol'].append(row[0])
				buy_period_dict['Buy_period'].append([current_period[0], current_period[-1]])
				current_period = []
		else:
			# Append the period dates here
			print(type(df.groupby(df['Symbol'])['Date'][i][1].values[j]))
			# current_period.append(df.groupby(df['Symbol'])['Date'][i].values[j])
new_df = pd.DataFrame(buy_period_dict)
new_df
# for i in df.groupby(df['Symbol'])['Date']:
# 	print(i[1].values[0])


#%%
