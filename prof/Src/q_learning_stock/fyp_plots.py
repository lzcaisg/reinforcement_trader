# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'eddy_src/q_learning_stock'))
	print(os.getcwd())
except:
	pass

import numpy as np
import pandas as pd
import indicators
import util
import config
import pred_model
import matplotlib.pyplot as plt
import datetime
import sys
from bokeh.io import curdoc, output_notebook, show
from bokeh.layouts import row, column, gridplot
from bokeh.models import ColumnDataSource, RangeTool, BoxAnnotation, HoverTool
from bokeh.models.widgets import Slider, TextInput
from bokeh.plotting import figure, output_file
from bokeh.core.properties import value

#%%
# Stock formed indexes vs actual index
def plot_passive_daily_comparisons(df_list: list, stock:str):
	"""**First dataframe must be the Portfolio with switching.** 
	"""
	temp_df1 = df_list[0].iloc[0:0]
	# temp_df1.drop(temp_df1.columns[0],axis=1,inplace=True)
	temp_df2 = df_list[1].iloc[0:0]
	# temp_df2.drop(temp_df2.columns[0],axis=1,inplace=True)
	temp_date_range = df_list[0]['Date'].tolist()
	for date in temp_date_range:
		df1 = df_list[0][df_list[0]['Date'] == date]
		# df1.drop(df1.columns[0],axis=1,inplace=True)
		# print(df1)

		# sys.exit()
		df2 = df_list[1][df_list[1]['Date'] == date]
		if not (df1.empty or df2.empty):
			temp_df1.append(df1, ignore_index=True)
			temp_df2.append(df2, ignore_index=True)

	# print(temp_df1)

	# sys.exit()
	p = figure(title="Daily price Comparison", x_axis_type='datetime', background_fill_color="#fafafa")

	p.add_tools(HoverTool(
			tooltips=[
				( 'Date', '@x{%F}'),
				( 'Price',  '$@y{%0.2f}'), # use @{ } for field names with spaces
			],
			formatters={
				'x': 'datetime', # use 'datetime' formatter for 'date' field,
				'y' : 'printf'
			},
			mode='mouse'
		))
	p.line(temp_df1['Date'].tolist(), temp_df1['Net'].values.tolist(), legend="Rebalanced stock portfolio",
		line_color="black")
	p.line(temp_df2['Date'].tolist(), temp_df2[stock].values.tolist(), legend=f"{stock} index")
	p.legend.location = "top_left"
	show(p)

stock_list = ['^BVSP', '^TWII', '^IXIC']
for symbol in stock_list:
	daily_df = pd.read_csv(f'data/algo/{symbol}/daily_nav.csv', parse_dates=['Date'])
	passive_daily_df = pd.read_csv('data/algo/index/passive_daily_nav.csv', parse_dates=['Date'])
	df_list = [daily_df, passive_daily_df]
	plot_passive_daily_comparisons(df_list, symbol)
	
#%%
df1 = pd.read_csv('data/goldman/GGSIX.csv')
df2 = pd.read_csv('data/goldman/GOIIX.csv')
df3 = pd.read_csv('data/goldman/GIPIX.csv')
new_df = pd.DataFrame({'GGSIX': df1['Close'].values, 'GOIIX': df2['Close'].values, 'GIPIX': df3['Close'].values}).corr()
new_df.to_csv('data/goldman/correlation_matrix.csv')

#%%
