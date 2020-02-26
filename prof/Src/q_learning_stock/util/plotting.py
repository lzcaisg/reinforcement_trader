import pandas as pd
import matplotlib.pyplot as plt
from bokeh.io import curdoc, output_notebook, show
from bokeh.layouts import row, column, gridplot
from bokeh.models import ColumnDataSource, RangeTool, BoxAnnotation, HoverTool
from bokeh.models.widgets import Slider, TextInput
from bokeh.plotting import figure, output_file
import config

def plot_loss(history):
    """ Plot the loss and val_loss
    """
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

def plot_pred(prediction, actual):
    """ Plot the loss and val_loss
    """
    plt.figure()
    plt.plot(prediction)
    plt.plot(actual)
    plt.title('Prediction')
    plt.ylabel('Amount')
    plt.xlabel('Time')
    plt.legend(['Prediction', 'Actual'], loc='upper left')
    plt.show()

def plot_acc(history):
    """ Plot accuracy
    """
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

def plot_market_trends(df: pd.DataFrame, indicators: dict, symbol=''):
    """Plots the market trend according to indicators

    Parameters
    --
    indicators: dict
        format should be {indicator: [buy_periods, sell_periods], ...}
    """    
    select = figure(title=symbol,
            plot_height=130, plot_width=800,
            x_axis_type="datetime", y_axis_type=None,
            tools="", toolbar_location=None, background_fill_color="#efefef")

    source = ColumnDataSource(data=dict(x=df.index, y=df[config.label_name]))
    select.line('x', 'y', source=source)
    select.ygrid.grid_line_color = None
    
    plots = []
    indicator_buy_periods = dict()
    
    for indicator in indicators:
        source = ColumnDataSource(data=dict(x=df.index, y=df[indicator]))

        p = figure(plot_height=300, plot_width=800,
            x_axis_type="datetime", x_axis_location="above", tools='pan,wheel_zoom,reset',
            background_fill_color="#efefef", x_range=select.x_range)

        p.line('x', 'y', source=source)
        p.yaxis.axis_label = 'Price'
        p.xaxis.axis_label = '{}: {}'.format(symbol, indicator)
        p.add_tools(HoverTool(
            tooltips=[
                ( 'date', '@x{%F}'),
                ( 'y',  '@y'), # use @{ } for field names with spaces
            ],
            formatters={
                'x': 'datetime', # use 'datetime' formatter for 'date' field
            },
            # display a tooltip whenever the cursor is vertically in line with a glyph
            mode='vline'
        ))
        for buy in indicators[indicator]['buy_periods']:
            p.add_layout(BoxAnnotation(left=buy[0], right =buy[1], fill_alpha=0.1, fill_color='green', line_color='green'))

        for sell in indicators[indicator]['sell_periods']:
            p.add_layout(BoxAnnotation(left=sell[0], right =sell[1], fill_alpha=0.1, fill_color='red', line_color='red'))
        # p.add_layout(BoxAnnotation(top=80, fill_alpha=0.1, fill_color='red', line_color='red'))

        plots.append(p)
        
        indicator_buy_periods[indicator] = indicators[indicator]['buy_periods']

    _print_holding_stats(indicator_buy_periods, df)
    show(column(plots))

def _print_holding_stats(periods: dict, df: pd.DataFrame):
    """Print stats for each period for:
    1. Number of holding periods 
    2. Average holding period
    3. Average return per period

    periods should be a dict of {indicator: [buy_period1, ...], ...}
    """
    for indicator in periods:
        num_hold_periods = len(periods[indicator])
        print("For {}: \n Number of holding periods = {}".format(indicator, str(num_hold_periods)))
        total_hold_period = 0
        total_return = 0
        for period in periods[indicator]:
            total_hold_period += period[1] - period[0]
            total_return += (df.loc[period[1]][config.label_name] - df.loc[period[0]][config.label_name])
        
        if num_hold_periods != 0:
            avg_hold_period = total_hold_period / num_hold_periods
            print("Average holding period = " + str(avg_hold_period) + " days")
            avg_return_per_period = total_return / num_hold_periods
            print("Average return per period = "+ str(avg_return_per_period))

