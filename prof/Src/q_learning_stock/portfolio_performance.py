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

import indicators
import util
import config
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from bokeh.io import curdoc, output_notebook, show
from bokeh.layouts import row, column, gridplot
from bokeh.models import ColumnDataSource, RangeTool, BoxAnnotation, HoverTool
from bokeh.models.widgets import Slider, TextInput
from bokeh.plotting import figure, output_file

#%%
stocks = ['GCIIX', 'GCGIX', 'GCVIX']#, 'GERIX', 'GSLC', 'GEM', 'GARSX', 'GCSIX', 'GSIE', 'GICIX', 'EWZ']
# stocks = ['GGSIX', 'GOIIX', 'GIPIX']
select_indicators = ['MACD Line']#, 'RSI', 'Stochastic Oscillator %K', 'CCI']
start = '1/1/2016'
end = '31/12/2018'
action_periods_dict = {}

for stock in stocks:
	df = pd.read_csv('data/goldman/{}.csv'.format(stock), usecols=config.column_names, parse_dates=['Date'])
	df = df.loc[(df['Date'] > start) & (df['Date'] <= end)]
	df = indicators.create_indicator_columns(df)
	indicator_dict = {}

	for indicator in select_indicators:
		indicator_dict[indicator] = indicators.get_action_periods(df, indicator)

	util.plot_market_trends(df, indicator_dict, stock)
	action_periods_dict[stock] = indicator_dict

#%% [markdown]
## Plot the switching portfolio performance for goldman sachs
#%%
portfolios = ['balanced', 'equity_growth', 'growth', 'growth_income', 'satellite']
df_list = []
# Add the switched portfolio first due to limitations by plotting (for now)
df_list.append(pd.read_csv('data/goldman/best_portfolio_switch.csv'))
for portfolio in portfolios:
	df = pd.read_csv('data/goldman/portfolio/{}/portfolio_quarterly_return.csv'.format(portfolio))
	# Reduce the df to 1 symbol only (for now)
	df_list.append(df.loc[:11])

#%%
def plot_portfolio_comparisons(df_list: list, date_col='end_period', price_col='quarterly_return', symbol_col='symbol'):
    """**First dataframe must be the Portfolio with switching.** 
    Currently only supports up to 6 dataframes in list.
    """
    x_min = datetime.datetime.strptime(df_list[0][date_col].min(), '%Y-%m-%d')
    x_max = datetime.datetime.strptime(df_list[0][date_col].max(), '%Y-%m-%d')
    y_min = df_list[0][price_col].min()
    y_max = df_list[0][price_col].max()

    # Temporary hack for date. Should have other date ranges
    date = []
    for i in df_list[0][date_col].values.tolist():
        date.append(datetime.datetime.strptime(i, '%Y-%m-%d'))

    p = figure(title="Portfolio Comparison",
           x_range=(x_min, x_max), y_range=(y_min, y_max), x_axis_type='datetime',
           background_fill_color="#fafafa")

    p.line(date, df_list[0][price_col].values.tolist(), legend="Portfolio with switching",
        line_color="black", line_dash="dashed")

    p.line(date, df_list[1][price_col].values.tolist(), legend="{}".format(df_list[1].iloc[0][symbol_col]))
    p.circle(date, df_list[1][price_col].values.tolist(), legend="{}".format(df_list[1].iloc[0][symbol_col]))

    p.line(date, df_list[2][price_col].values.tolist(), legend="{}".format(df_list[2].iloc[0][symbol_col]), line_color="olivedrab")
    p.circle(date, df_list[2][price_col].values.tolist(), legend="{}".format(df_list[2].iloc[0][symbol_col]),
            fill_color=None, line_color="olivedrab")

    p.line(date, df_list[3][price_col].values.tolist(), legend="{}".format(df_list[3].iloc[0][symbol_col]),
        line_color="gold", line_width=2)

    p.line(date, df_list[4][price_col].values.tolist(), legend="{}".format(df_list[4].iloc[0][symbol_col]),
        line_dash="dotted", line_color="indigo", line_width=2)

    p.line(date, df_list[5][price_col].values.tolist(), legend="{}".format(df_list[5].iloc[0][symbol_col]),
        line_color="coral", line_dash="dotdash", line_width=2)

    p.legend.location = "bottom_left"

    # output_file("data/portfolio_comparison.html", title="Portfolio Comparison")

    show(p)
plot_portfolio_comparisons(df_list)

#%% [markdown]
## Plot the switching portfolio performance according to 8 quarters

#%%
def plot_portfolio_8q_comparisons(df_list: list, date_col='end_period', price_col='quarterly_return', symbol_col='symbol'):

    # Temporary hack for date. Should have other date ranges
    # create 3 date ranges and convert them to datetime
    date = []
    for i in range(3):
        end_date = df_list[0][date_col].iloc[i*2+7]
        date.append(datetime.datetime.strptime(end_date, '%Y-%m-%d'))
    print(date)

    return_range = []
    for i in range(6):
        return_range.append([])
        for j in range(3):
            returns = (df_list[i][price_col].iloc[j*2+7]+df_list[i][price_col].iloc[j*2])/2
            return_range[i].append(returns)
    print(return_range)

    p = figure(title="Portfolio Comparison for 8 Quarters starting from 1 jan 2016",
           x_range=(date[0], date[2]), y_range=(-10, 10), x_axis_type='datetime',
           background_fill_color="#fafafa")

    p.line(date, return_range[0], legend="Portfolio with switching",
        line_color="black", line_dash="dashed")

    p.line(date, return_range[1], legend="{}".format(df_list[1].iloc[0][symbol_col]))
    p.circle(date, return_range[1], legend="{}".format(df_list[1].iloc[0][symbol_col]))

    p.line(date, return_range[2], legend="{}".format(df_list[2].iloc[0][symbol_col]), line_color="olivedrab")
    p.circle(date, return_range[2], legend="{}".format(df_list[2].iloc[0][symbol_col]),
            fill_color=None, line_color="olivedrab")

    p.line(date, return_range[3], legend="{}".format(df_list[3].iloc[0][symbol_col]),
        line_color="gold", line_width=2)

    p.line(date, return_range[4], legend="{}".format(df_list[4].iloc[0][symbol_col]),
        line_dash="dotted", line_color="indigo", line_width=2)

    p.line(date, return_range[5], legend="{}".format(df_list[5].iloc[0][symbol_col]),
        line_color="coral", line_dash="dotdash", line_width=2)

    p.legend.location = "top_right"

    # output_file("data/portfolio_comparison.html", title="Portfolio Comparison")

    show(p)
    # save_df = pd.DataFrame({'date': date, 'returns': return_range[0]})
    # save_df.to_csv('data/8_quarterly.csv')

#%%
plot_portfolio_8q_comparisons(df_list)

#%%
