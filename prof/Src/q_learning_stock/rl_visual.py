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
from bokeh.models import ColumnDataSource, RangeTool, BoxAnnotation, HoverTool, BoxAnnotation
from bokeh.models.widgets import Slider, TextInput
from bokeh.plotting import figure, output_file, save
from bokeh.core.properties import value

#%%
# Index sampled
df = pd.read_csv('data/rl/index_sampled/daily_nav.csv', parse_dates=['Date'])
passive_df = pd.read_csv('data/rl/index_sampled/passive_daily_nav.csv', parse_dates=['Date'])
actions_df = pd.read_csv('data/rl/index_sampled/actions_taken.csv', parse_dates=['Date'])
df_list = [df, passive_df, actions_df]

def plot_daily_nav(df_list: list, x_col='Date'):
    """**First dataframe must be the Portfolio with switching.** 
    """
    p = figure(title="Portfolio Net Asset Comparison", x_axis_type='datetime',
                background_fill_color="#fafafa")

    p.line(df_list[0][x_col], df_list[0]['Net'].values.tolist(), legend="RL rebalanced",
        line_color="black")

    p.line(df_list[1][x_col], df_list[1]["^BVSP"].values.tolist(), legend="^BVSP",
        line_color="red")

    p.line(df_list[1][x_col], df_list[1]["^TWII"].values.tolist(), legend="^TWII", 
        line_color="orange")

    p.line(df_list[1][x_col], df_list[1]["^IXIC"].values.tolist(), legend="^IXIC",
        line_color="olivedrab")

    # colours = ['red', 'yellow', 'blue', 'green', 'black']
    # for i in range(len(df_list[2])):
    #     box = BoxAnnotation(left=df_list[2].iloc[i]['Date'], right=df_list[2].iloc[i]['Date']+ datetime.timedelta(days=1), fill_alpha=0.4, 
    #                         fill_color=colours[df_list[2].iloc[i]['Action']])
    #     p.add_layout(box)

    p.legend.location = "top_left"
    output_file('data/rl/index_sampled/daily_nav_comp.html')
    save(p)
    show(p)
plot_daily_nav(df_list)

#%%
# Index
lagged = False

if not lagged:
    df = pd.read_csv('data/rl/index/daily_nav.csv', parse_dates=['Date'])
    passive_df = pd.read_csv('data/rl/index/passive_daily_nav.csv', parse_dates=['Date'])
    actions_df = pd.read_csv('data/rl/index/actions_taken.csv', parse_dates=['Date'])
    df_list = [df, passive_df, actions_df]
else:
    df = pd.read_csv('data/rl/index/lagged/daily_nav.csv', parse_dates=['Date'])
    passive_df = pd.read_csv('data/rl/index/passive_daily_nav.csv', parse_dates=['Date'])
    actions_df = pd.read_csv('data/rl/index/lagged/actions_taken.csv', parse_dates=['Date'])
    df_list = [df, passive_df, actions_df]

def plot_daily_nav(df_list: list, x_col='Date'):
    """**First dataframe must be the Portfolio with switching.** 
    """
    p = figure(title="Portfolio Net Asset Comparison", x_axis_type='datetime',
                background_fill_color="#fafafa")

    p.line(df_list[0][x_col], df_list[0]['Net'].values.tolist(), legend="RL rebalanced",
        line_color="black")

    p.line(df_list[1][x_col], df_list[1]["^BVSP"].values.tolist(), legend="^BVSP",
        line_color="red")

    p.line(df_list[1][x_col], df_list[1]["^TWII"].values.tolist(), legend="^TWII", 
        line_color="orange")

    p.line(df_list[1][x_col], df_list[1]["^IXIC"].values.tolist(), legend="^IXIC",
        line_color="olivedrab")

    colours = ['red', 'yellow', 'green', '']
    # for i in range(len(df_list[2])):
    #     box = BoxAnnotation(left=df_list[2].iloc[i]['Date'], right=df_list[2].iloc[i]['Date']+ datetime.timedelta(days=1), fill_alpha=0.4, 
    #                         fill_color=colours[df_list[2].iloc[i]['Action']])
    #     p.add_layout(box)

    p.legend.location = "top_left"
    output_file('data/rl/index/daily_nav_comp.html')
    save(p)
    show(p)

plot_daily_nav(df_list)

#%%
# BVSP
df = pd.read_csv('data/rl/^BVSP/daily_nav.csv', parse_dates=['Date'])
passive_df = pd.read_csv('data/rl/^BVSP/passive_daily_nav.csv', parse_dates=['Date'])
actions_df = pd.read_csv('data/rl/^BVSP/actions_taken.csv', parse_dates=['Date'])
df_list = [df, passive_df, actions_df]

def plot_daily_nav(df_list: list, x_col='Date'):
    """**First dataframe must be the Portfolio with switching.** 
    """
    p = figure(title="Portfolio Net Asset Comparison", x_axis_type='datetime',
                background_fill_color="#fafafa")

    p.line(df_list[0][x_col], df_list[0]['Net'].values.tolist(), legend="RL rebalanced",
        line_color="black")

    p.line(df_list[1][x_col], df_list[1]["EQTL3.SA"].values.tolist(), legend="EQTL3.SA")

    p.line(df_list[1][x_col], df_list[1]["ITSA4.SA"].values.tolist(), legend="ITSA4.SA", line_color="olivedrab")

    p.line(df_list[1][x_col], df_list[1]["PETR3.SA"].values.tolist(), legend="PETR3.SA",
        line_color="orange")

    colours = ['red', 'yellow', 'blue', 'green', 'black']
    for i in range(len(df_list[2])):
        box = BoxAnnotation(left=df_list[2].iloc[i]['Date'], right=df_list[2].iloc[i]['Date']+ datetime.timedelta(days=1), fill_alpha=0.4, 
                            fill_color=colours[df_list[2].iloc[i]['Action']])
        p.add_layout(box)

    p.legend.location = "top_left"
    output_file('data/rl/^BVSP/daily_nav_comp.html')
    save(p)
    show(p)

plot_daily_nav(df_list)