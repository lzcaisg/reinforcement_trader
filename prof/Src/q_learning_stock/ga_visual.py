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
import matplotlib.pyplot as plt
import datetime
from bokeh.io import curdoc, output_notebook, show
from bokeh.layouts import row, column, gridplot
from bokeh.models import ColumnDataSource, RangeTool, BoxAnnotation, HoverTool
from bokeh.models.widgets import Slider, TextInput
from bokeh.plotting import figure, output_file
from bokeh.core.properties import value

#%%
run_set = ['goldman', 'index', '^BVSP', '^TWII', '^IXIC', 'index_sampled']
choose_set = run_set[5]
base_rate = True

#%%
if choose_set == run_set[0]:
    if base_rate:
        ga_algo_df = pd.read_csv('data/algo/goldman/daily_nav.csv')
        df = pd.read_csv('data/algo/goldman/passive_daily_nav.csv')
        high_risk_df = df['GGSIX']
        med_risk_df = df['GOIIX']
        low_risk_df = df['GIPIX']
        df_list = [ga_algo_df, high_risk_df, med_risk_df, low_risk_df]
    else:
        ga_algo_df = pd.read_csv('data/algo/goldman/no_br/daily_nav.csv')
        df = pd.read_csv('data/algo/goldman/passive_daily_nav.csv')
        high_risk_df = df['GGSIX']
        med_risk_df = df['GOIIX']
        low_risk_df = df['GIPIX']
        df_list = [ga_algo_df, high_risk_df, med_risk_df, low_risk_df]

    def plot_portfolio_comparisons(df_list: list, x_col='Date'):
        """**First dataframe must be the Portfolio with switching.** 
        """
        x_min = datetime.datetime.strptime(df_list[0][x_col].min(), '%Y-%m-%d')
        x_max = datetime.datetime.strptime(df_list[0][x_col].max(), '%Y-%m-%d')

        # Temporary hack for date. Should have other date ranges
        date = []
        for i in df_list[0][x_col].values.tolist():
            date.append(datetime.datetime.strptime(i, '%Y-%m-%d'))

        p = figure(title="Portfolio Net Asset Comparison",
            x_range=(x_min, x_max), x_axis_type='datetime',
            background_fill_color="#fafafa")

        p.line(date, df_list[0]['Net'].values.tolist(), legend="Reallocating portfolio",
            line_color="black")

        p.line(date, df_list[1].values.tolist(), legend="GGSIX", line_color="red")

        p.line(date, df_list[2].values.tolist(), legend="GOIIX", line_color="orange")

        p.line(date, df_list[3].values.tolist(), legend="GIPIX", line_color="olivedrab")

        p.legend.location = "top_left"

        show(p)
    plot_portfolio_comparisons(df_list)

    def plot_daily_comp_comparisons(df: pd.DataFrame):
        date_range = df['Date'].tolist()
        nav_list = np.array(df['Net'].values.tolist())

        p = figure(title="GS Daily NAV Composition",
            x_axis_type='datetime',
            background_fill_color="#fafafa")

        p.add_tools(HoverTool(
                tooltips=[
                    ( 'Date', '@x{%F}'),
                    ( 'Percentage',  '@y{%0.2f}%'), # use @{ } for field names with spaces
                ],
                formatters={
                    'x': 'datetime', # use 'datetime' formatter for 'date' field,
                    'y' : 'printf'
                },
                mode='mouse'
            ))

        p.line(date_range, np.array(df['GGSIX'].values.tolist())/nav_list*100, legend="GGSIX", line_color="red")

        p.line(date_range, np.array(df['GOIIX'].values.tolist())/nav_list*100, legend="GOIIX", 
            line_color="orange")

        p.line(date_range, np.array(df['GIPIX'].values.tolist())/nav_list*100, legend="GIPIX",
            line_color="olivedrab")
        
        # for date in date_range:
        #     if df[df['Date'] == date]['Adjusted'].values[0] == True:
        #         p.add_layout(BoxAnnotation(left=date, right=date + datetime.timedelta(days=1), fill_alpha=0.1, fill_color='green', line_color='green'))
                
        p.legend.location = "top_left"

        show(p)

    plot_daily_comp_comparisons(ga_algo_df)

#%% [markdown]
# # Index NAV
#%%
if choose_set == run_set[1]:
    if base_rate:
        index_daily_df = pd.read_csv('data/algo/index/daily_nav.csv', parse_dates=['Date'])
        passive_index_daily_df = pd.read_csv('data/algo/index/passive_daily_nav.csv', parse_dates=['Date'])
        df_list = [index_daily_df, passive_index_daily_df]
    else:
        index_daily_df = pd.read_csv('data/algo/index/no_br/daily_nav.csv', parse_dates=['Date'])
        passive_index_daily_df = pd.read_csv('data/algo/index/passive_daily_nav.csv', parse_dates=['Date'])
        df_list = [index_daily_df, passive_index_daily_df]

    def plot_daily_comparisons(df_list: list, x_col='Date'):
        """**First dataframe must be the Portfolio with switching.** 
        """
        date_range = df_list[0]['Date'].tolist()
        x_min = date_range[0]
        x_max = date_range[-1]
        y_min = df_list[0]['Net'].min()
        y_max = df_list[0]['Net'].max()

        p = figure(title="Index Daily NAV Comparison",
            x_range=(x_min, x_max), y_range=(y_min, y_max), x_axis_type='datetime',
            background_fill_color="#fafafa")

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

        p.line(date_range, df_list[0]['Net'].values.tolist(), legend="Reallocating portfolio",
            line_color="black")

        p.line(date_range, df_list[1]['^BVSP'].values.tolist(), legend="{}".format('^BVSP'),
            line_color="red")

        p.line(date_range, df_list[1]['^TWII'].values.tolist(), legend="{}".format('^TWII'), 
            line_color="orange")

        p.line(date_range, df_list[1]['^IXIC'].values.tolist(), legend="{}".format('^IXIC'),
            line_color="olivedrab")

        # for date in date_range:
        #     if df_list[0][df_list[0]['Date'] == date]['Adjusted'].values[0] == True:
        #         p.add_layout(BoxAnnotation(left=date, right=date + datetime.timedelta(days=1), fill_alpha=0.1, fill_color='green', line_color='green'))

        p.legend.location = "top_left"

        show(p)

    plot_daily_comparisons(df_list)

    # def plot_daily_comp_comparisons(df: pd.DataFrame):
    #     date_range = df['Date'].tolist()
    #     x_min = date_range[0]
    #     x_max = date_range[-1]
    #     y_min = 0
    #     y_max = 100
    #     nav_list = np.array(df['Net'].values.tolist())

    #     p = figure(title="Index Daily NAV Composition",
    #         x_range=(x_min, x_max), y_range=(y_min, y_max), x_axis_type='datetime',
    #         background_fill_color="#fafafa")

    #     p.add_tools(HoverTool(
    #             tooltips=[
    #                 ( 'Date', '@x{%F}'),
    #                 ( 'Percentage',  '@y{%0.2f}%'), # use @{ } for field names with spaces
    #             ],
    #             formatters={
    #                 'x': 'datetime', # use 'datetime' formatter for 'date' field,
    #                 'y' : 'printf'
    #             },
    #             mode='mouse'
    #         ))
        
    #     colors = ["blue", "olivedrab", "orange"]
    #     stocks = ['^BVSP', '^TWII', '^IXIC']
    #     data = {'composition': date_range,
    #             '^BVSP': np.array(df['^BVSP'].values.tolist())/nav_list*100,
    #             '^TWII': np.array(df['^TWII'].values.tolist())/nav_list*100,
    #             '^IXIC': np.array(df['^IXIC'].values.tolist())/nav_list*100}
        
    #     p.vbar_stack(stocks, x='composition', width=0.1, color=colors, source=data, legend=[value(x) for x in stocks])

    #     p.legend.location = "top_left"

    #     show(p)

    # plot_daily_comp_comparisons(index_daily_df)

    def plot_daily_comp_comparisons(df: pd.DataFrame):
        date_range = df['Date'].tolist()
        x_min = date_range[0]
        x_max = date_range[-1]
        y_min = 0
        y_max = 100
        nav_list = np.array(df['Net'].values.tolist())

        p = figure(title="Index Daily NAV Composition",
            x_range=(x_min, x_max), y_range=(y_min, y_max), x_axis_type='datetime',
            background_fill_color="#fafafa")

        p.add_tools(HoverTool(
                tooltips=[
                    ( 'Date', '@x{%F}'),
                    ( 'Percentage',  '@y{%0.2f}%'), # use @{ } for field names with spaces
                ],
                formatters={
                    'x': 'datetime', # use 'datetime' formatter for 'date' field,
                    'y' : 'printf'
                },
                mode='mouse'
            ))

        p.line(date_range, np.array(df['^BVSP'].values.tolist())/nav_list*100, legend="{}".format('^BVSP'),
            line_color="red")

        p.line(date_range, np.array(df['^TWII'].values.tolist())/nav_list*100, legend="{}".format('^TWII'), 
            line_color="orange")

        p.line(date_range, np.array(df['^IXIC'].values.tolist())/nav_list*100, legend="{}".format('^IXIC'),
            line_color="olivedrab")
        
        # for date in date_range:
        #     if df[df['Date'] == date]['Adjusted'].values[0] == True:
        #         p.add_layout(BoxAnnotation(left=date, right=date + datetime.timedelta(days=1), fill_alpha=0.1, fill_color='green', line_color='green'))
                
        p.legend.location = "top_left"

        show(p)

    plot_daily_comp_comparisons(index_daily_df)

#%% [markdown]
# # Index components nav
#%%
if choose_set == run_set[2] or choose_set == run_set[3] or choose_set == run_set[4]:
    folders = {'^BVSP': ['EQTL3.SA', 'ITSA4.SA', 'PETR3.SA'], 
                '^TWII': ['1326.TW', '2882.TW', '3008.TW'], 
                '^IXIC': ['TSLA', 'IBKC', 'FEYE']}

    if base_rate:
        daily_df = pd.read_csv('data/algo/{}/daily_nav.csv'.format(choose_set), parse_dates=['Date'])
        passive_daily_df = pd.read_csv('data/algo/{}/passive_daily_nav.csv'.format(choose_set), parse_dates=['Date'])
        df_list = [daily_df, passive_daily_df]
    else:
        daily_df = pd.read_csv('data/algo/{}/no_br/daily_nav.csv'.format(choose_set), parse_dates=['Date'])
        passive_daily_df = pd.read_csv('data/algo/{}/passive_daily_nav.csv'.format(choose_set), parse_dates=['Date'])
        df_list = [daily_df, passive_daily_df]

    def plot_daily_comparisons(df_list: list, stock_list: list):
        """**First dataframe must be the Portfolio with switching.** 
        """
        date_range = df_list[0]['Date'].tolist()
        x_min = date_range[0]
        x_max = date_range[-1]
        y_min = df_list[0]['Net'].min()
        y_max = df_list[0]['Net'].max()

        p = figure(title="Daily NAV Comparison",
            x_range=(x_min, x_max), y_range=(y_min, y_max), x_axis_type='datetime',
            background_fill_color="#fafafa")

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

        p.line(date_range, df_list[0]['Net'].values.tolist(), legend="Reallocating portfolio",
            line_color="black")

        p.line(date_range, df_list[1][stock_list[0]].values.tolist(), legend="{}".format(stock_list[0]),
            line_color="red")

        p.line(date_range, df_list[1][stock_list[1]].values.tolist(), legend="{}".format(stock_list[1]), 
            line_color="orange")

        p.line(date_range, df_list[1][stock_list[2]].values.tolist(), legend="{}".format(stock_list[2]),
            line_color="olivedrab")

        # for date in date_range:
        #     if df_list[0][df_list[0]['Date'] == date]['Adjusted'].values[0] == True:
        #         p.add_layout(BoxAnnotation(left=date, right=date + datetime.timedelta(days=1), fill_alpha=0.1, fill_color='green', line_color='green'))

        p.legend.location = "top_left"

        show(p)

    plot_daily_comparisons(df_list, folders[choose_set])

    # def plot_daily_comp_comparisons(df: pd.DataFrame, stock_list: list):
    #     date_range = df['Date'].tolist()
    #     x_min = date_range[0]
    #     x_max = date_range[-1]
    #     y_min = 0
    #     y_max = 100
    #     nav_list = np.array(df['Net'].values.tolist())

    #     p = figure(title="Stocks Daily NAV Composition",
    #         x_range=(x_min, x_max), y_range=(y_min, y_max), x_axis_type='datetime',
    #         background_fill_color="#fafafa")

    #     p.add_tools(HoverTool(
    #             tooltips=[
    #                 ( 'Date', '@x{%F}'),
    #                 ( 'Percentage',  '@y{%0.2f}%'), # use @{ } for field names with spaces
    #             ],
    #             formatters={
    #                 'x': 'datetime', # use 'datetime' formatter for 'date' field,
    #                 'y' : 'printf'
    #             },
    #             mode='mouse'
    #         ))
        
    #     colors = ["blue", "olivedrab", "orange"]
    #     stocks = [stock_list[0], stock_list[1], stock_list[2]]
    #     data = {'composition': date_range,
    #             stock_list[0]: np.array(df[stock_list[0]].values.tolist())/nav_list*100,
    #             stock_list[1]: np.array(df[stock_list[1]].values.tolist())/nav_list*100,
    #             stock_list[2]: np.array(df[stock_list[2]].values.tolist())/nav_list*100}
        
    #     p.vbar_stack(stocks, x='composition', width=0.1, color=colors, source=data, legend=[value(x) for x in stocks])

    #     p.legend.location = "top_left"

    #     show(p)

    # plot_daily_comp_comparisons(daily_df, folders[choose_set])

    def plot_daily_comp_comparisons(df: pd.DataFrame, stock_list: list):
        date_range = df['Date'].tolist()
        x_min = date_range[0]
        x_max = date_range[-1]
        y_min = 0
        y_max = 100
        nav_list = np.array(df['Net'].values.tolist())

        p = figure(title="Daily NAV Composition Comparison",
            x_range=(x_min, x_max), y_range=(y_min, y_max), x_axis_type='datetime',
            background_fill_color="#fafafa")

        p.add_tools(HoverTool(
                tooltips=[
                    ( 'Date', '@x{%F}'),
                    ( 'Percentage',  '@y{%0.2f}%'), # use @{ } for field names with spaces
                ],
                formatters={
                    'x': 'datetime', # use 'datetime' formatter for 'date' field,
                    'y' : 'printf'
                },
                mode='mouse'
            ))

        p.line(date_range, np.array(df[stock_list[0]].values.tolist())/nav_list*100, legend="{}".format(stock_list[0]),
            line_color="red")

        p.line(date_range, np.array(df[stock_list[1]].values.tolist())/nav_list*100, legend="{}".format(stock_list[1]), 
            line_color="orange")

        p.line(date_range, np.array(df[stock_list[2]].values.tolist())/nav_list*100, legend="{}".format(stock_list[2]),
            line_color="olivedrab")
        
        # for date in date_range:
        #     if df[df['Date'] == date]['Adjusted'].values[0] == True:
        #         p.add_layout(BoxAnnotation(left=date, right=date + datetime.timedelta(days=1), fill_alpha=0.1, fill_color='green', line_color='green'))
                
        p.legend.location = "top_left"

        show(p)
    plot_daily_comp_comparisons(daily_df, folders[choose_set])

#%% [markdown]
# # Reallocated index components nav
#%%
if choose_set == run_set[5]:
    folders = ['index_sampled', '^BVSP', '^TWII', '^IXIC']

    if base_rate:
        df_list = []
        for i in range(len(folders)):
            df = pd.read_csv('data/algo/{}/daily_price.csv'.format(folders[i]), parse_dates=['Date'])
            df_list.append(df)
    else:
        df_list = []
        for i in range(len(folders)):
            df = pd.read_csv('data/algo/{}/no_br/daily_price.csv'.format(folders[i]), parse_dates=['Date'])
            df_list.append(df)

    date_range = util.remove_uncommon_dates(df_list)

    def plot_daily_comparisons(df_list: list, stock_list: list, date_range):
        """**First dataframe must be the Portfolio with switching.** 
        """

        p = figure(title="Daily price Comparison",
            x_axis_type='datetime',
            background_fill_color="#fafafa")

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

        p.line(date_range, df_list[0]['Close'].values.tolist(), legend="Reallocating portfolio",
            line_color="black")

        p.line(date_range, df_list[1]['Close'].values.tolist(), legend="{}".format(stock_list[1]),
            line_color="red")

        p.line(date_range, df_list[2]['Close'].values.tolist(), legend="{}".format(stock_list[2]), 
            line_color="orange")

        p.line(date_range, df_list[3]['Close'].values.tolist(), legend="{}".format(stock_list[3]),
            line_color="olivedrab")

        # for date in date_range:
        #     if df_list[0][df_list[0]['Date'] == date]['Adjusted'].values[0] == True:
        #         p.add_layout(BoxAnnotation(left=date, right=date + datetime.timedelta(days=1), fill_alpha=0.1, fill_color='green', line_color='green'))

        p.legend.location = "top_left"

        show(p)

    plot_daily_comparisons(df_list, folders, date_range)

    if base_rate:
        daily_df = pd.read_csv('data/algo/{}/daily_nav.csv'.format(choose_set), parse_dates=['Date'])
        passive_daily_df = pd.read_csv('data/algo/index/passive_daily_nav.csv', parse_dates=['Date'])
        df_list = [daily_df, passive_daily_df]
    else:
        daily_df = pd.read_csv('data/algo/{}/no_br/daily_nav.csv'.format(choose_set), parse_dates=['Date'])
        passive_daily_df = pd.read_csv('data/algo/index/passive_daily_nav.csv', parse_dates=['Date'])
        df_list = [daily_df, passive_daily_df]

    date_range = util.remove_uncommon_dates(df_list)

    # Compare with original index
    def plot_passive_daily_comparisons(df_list: list, stock_list: list, date_range):
        """**First dataframe must be the Portfolio with switching.** 
        """

        p = figure(title="Daily price Comparison",
            x_axis_type='datetime',
            background_fill_color="#fafafa")

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

        p.line(date_range, df_list[0]['Net'].values.tolist(), legend="Reallocating portfolio",
            line_color="black")

        p.line(date_range, df_list[1][stock_list[1]].values.tolist(), legend="{}".format(stock_list[1]),
            line_color="red")

        p.line(date_range, df_list[1][stock_list[2]].values.tolist(), legend="{}".format(stock_list[2]), 
            line_color="orange")

        p.line(date_range, df_list[1][stock_list[3]].values.tolist(), legend="{}".format(stock_list[3]),
            line_color="olivedrab")

        # for date in date_range:
        #     if df_list[0][df_list[0]['Date'] == date]['Adjusted'].values[0] == True:
        #         p.add_layout(BoxAnnotation(left=date, right=date + datetime.timedelta(days=1), fill_alpha=0.1, fill_color='green', line_color='green'))

        p.legend.location = "top_left"

        show(p)
    
    plot_passive_daily_comparisons(df_list, folders, date_range)

    # def plot_daily_comp_comparisons(df: pd.DataFrame):
    #     date_range = df['Date'].tolist()
    #     x_min = date_range[0]
    #     x_max = date_range[-1]
    #     y_min = 0
    #     y_max = 100
    #     nav_list = np.array(df['Net'].values.tolist())

    #     p = figure(title="Index Daily NAV Composition",
    #         x_range=(x_min, x_max), y_range=(y_min, y_max), x_axis_type='datetime',
    #         background_fill_color="#fafafa")

    #     p.add_tools(HoverTool(
    #             tooltips=[
    #                 ( 'Date', '@x{%F}'),
    #                 ( 'Percentage',  '@y{%0.2f}%'), # use @{ } for field names with spaces
    #             ],
    #             formatters={
    #                 'x': 'datetime', # use 'datetime' formatter for 'date' field,
    #                 'y' : 'printf'
    #             },
    #             mode='mouse'
    #         ))
        
    #     colors = ["blue", "olivedrab", "orange"]
    #     stocks = ['^BVSP', '^TWII', '^IXIC']
    #     data = {'composition': date_range,
    #             '^BVSP': np.array(df['^BVSP'].values.tolist())/nav_list*100,
    #             '^TWII': np.array(df['^TWII'].values.tolist())/nav_list*100,
    #             '^IXIC': np.array(df['^IXIC'].values.tolist())/nav_list*100}
        
    #     p.vbar_stack(stocks, x='composition', width=0.1, color=colors, source=data, legend=[value(x) for x in stocks])

    #     p.legend.location = "top_left"

    #     show(p)

    # plot_daily_comp_comparisons(daily_df)

    def plot_daily_comp_comparisons(df: pd.DataFrame, stock_list: list, date_range):
        date_range = df['Date'].tolist()
        nav_list = np.array(df['Net'].values.tolist())
        x_min = date_range[0]
        x_max = date_range[-1]
        y_min = 0
        y_max = 100

        p = figure(title="Daily NAV Composition Comparison",
            x_range=(x_min, x_max), y_range=(y_min,y_max), x_axis_type='datetime',
            background_fill_color="#fafafa")

        p.add_tools(HoverTool(
                tooltips=[
                    ( 'Date', '@x{%F}'),
                    ( 'Percentage',  '@y{%0.2f}%'), # use @{ } for field names with spaces
                ],
                formatters={
                    'x': 'datetime', # use 'datetime' formatter for 'date' field,
                    'y' : 'printf'
                },
                mode='mouse'
            ))

        p.line(date_range, np.array(df[stock_list[1]].values.tolist())/nav_list*100, legend="{}".format(stock_list[1]),
            line_color="red")

        p.line(date_range, np.array(df[stock_list[2]].values.tolist())/nav_list*100, legend="{}".format(stock_list[2]), 
            line_color="orange")

        p.line(date_range, np.array(df[stock_list[3]].values.tolist())/nav_list*100, legend="{}".format(stock_list[3]),
            line_color="olivedrab")
        
        # for date in date_range:
        #     if df[df['Date'] == date]['Adjusted'].values[0] == True:
        #         p.add_layout(BoxAnnotation(left=date, right=date + datetime.timedelta(days=1), fill_alpha=0.1, fill_color='green', line_color='green'))
                
        p.legend.location = "top_left"

        show(p)
    plot_daily_comp_comparisons(daily_df, folders, date_range)