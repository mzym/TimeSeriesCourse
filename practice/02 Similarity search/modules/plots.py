import numpy as np
import pandas as pd

# for visualization
import plotly
from plotly.subplots import make_subplots
from plotly.offline import init_notebook_mode
import plotly.graph_objs as go
import plotly.express as px
plotly.offline.init_notebook_mode(connected=True)


def plot_ts_set(ts_set, title='Input Time Series Set'):
    """
    Plot the time series set.

    Parameters
    ----------
    ts_set : numpy.ndarrray (2d array of shape (ts_number, ts_length))
        Time series set.
    
    title : str, default = 'Input Time Series Set'
        Title of plot.
    """

    ts_num, m = ts_set.shape

    fig = go.Figure()

    for i in range(ts_num):
        fig.add_trace(go.Scatter(x=np.arange(m), y=ts_set[i], line=dict(width=3), name="Time series " + str(i)))

    fig.update_xaxes(showgrid=False,
                     title='Time',
                     title_font=dict(size=22, color='black'),
                     linecolor='#000',
                     ticks="outside",
                     tickfont=dict(size=18, color='black'),
                     linewidth=2,
                     tickwidth=2)
    fig.update_yaxes(showgrid=False,
                     title='Values',
                     title_font=dict(size=22, color='black'),
                     linecolor='#000',
                     ticks="outside",
                     tickfont=dict(size=18, color='black'),
                     zeroline=False,
                     linewidth=2,
                     tickwidth=2)

    fig.update_layout(title=title,
                      title_font=dict(size=24, color='black'),
                      plot_bgcolor="rgba(0,0,0,0)",
                      paper_bgcolor='rgba(0,0,0,0)',
                      legend=dict(font=dict(size=20, color='black'))
                      )

    fig.show(renderer="colab")


def plot2d(x, y, plot_title, x_title, y_title):
    """
    2D Plot on figure for different experiments.

    Parameters
    ----------
    x : numpy.ndarrray
        Values of x axis of plot.

    y : numpy.ndarrray
        Values of y axis of plot.
    
    plot_title : str
        Title of plot.

    x_title : str
        Title of x axis of plot.

    y_title : str
        Title of y axis of plot.
    """

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=x, y=y))

    fig.update_xaxes(showgrid=False,
                     title=x_title,
                     title_font=dict(size=22, color='black'),
                     linecolor='#000',
                     ticks="outside",
                     tickfont=dict(size=18, color='black'),
                     linewidth=2,
                     tickwidth=2,
                     mirror=True)
    fig.update_yaxes(showgrid=False,
                     title=y_title,
                     title_font=dict(size=22, color='black'),
                     linecolor='#000',
                     ticks="outside",
                     tickfont=dict(size=18, color='black'),
                     zeroline=False,
                     linewidth=2,
                     tickwidth=2,
                     mirror=True)

    fig.update_layout(title={'text': plot_title, 'x': 0.5, 'xanchor': 'center'},
                      title_font=dict(size=24, color='black'),
                      plot_bgcolor="rgba(0,0,0,0)",
                      paper_bgcolor='rgba(0,0,0,0)',
                      width=700)

    fig.show(renderer="colab")


def mplot2d(x, y, plot_title=None, x_title=None, y_title=None, trace_titles=None):
    """
    Multiple 2D Plots on figure for different experiments.

    Parameters
    ----------
    x : numpy.ndarrray
        Values of x axis of plot.

    y : numpy.ndarrray
        Values of y axis of plot.
    
    plot_title : str
        Title of plot.

    x_title : str
        Title of x axis of plot.

    y_title : str
        Title of y axis of plot.

    trace_titles : numpy.ndarrray
        Titles of plots.
    """

    fig = go.Figure()

    for i in range(y.shape[0]):
        fig.add_trace(go.Scatter(x=x, y=y[i], line=dict(width=3), name=trace_titles[i]))

    fig.update_xaxes(showgrid=False,
                     title=x_title,
                     title_font=dict(size=22, color='black'),
                     linecolor='#000',
                     ticks="outside",
                     tickfont=dict(size=18, color='black'),
                     linewidth=2,
                     tickwidth=2,
                     tickvals=x)
    fig.update_yaxes(showgrid=False,
                     title=y_title,
                     title_font=dict(size=22, color='black'),
                     linecolor='#000',
                     ticks="outside",
                     tickfont=dict(size=18, color='black'),
                     zeroline=False,
                     linewidth=2,
                     tickwidth=2)

    fig.update_layout(title={'text': plot_title, 'x': 0.5, 'xanchor': 'center'},
                      title_font=dict(size=24, color='black'),
                      plot_bgcolor="rgba(0,0,0,0)",
                      paper_bgcolor='rgba(0,0,0,0)',
                      legend=dict(font=dict(size=20, color='black')),
                      width=1000,
                      height=600
                      )

    fig.show(renderer="colab")


def plot_bestmatch_data(ts, query):
    """
    Visualize the input data (time series and query) for the best match task.

    Parameters
    ----------
    ts : numpy.ndarrray
        Time series.

    query : numpy.ndarrray
        Query.
    """

    query_len = query.shape[0]
    ts_len = ts.shape[0]

    fig = make_subplots(rows=1, cols=2, column_widths=[0.1, 0.9], subplot_titles=("Query", "Time Series"), horizontal_spacing=0.04)

    fig.add_trace(go.Scatter(x=np.arange(query_len), y=query, line=dict(color=px.colors.qualitative.Plotly[1])),
                row=1, col=1)
    fig.add_trace(go.Scatter(x=np.arange(ts_len), y=ts, line=dict(color=px.colors.qualitative.Plotly[0])),
                row=1, col=2)

    fig.update_annotations(font=dict(size=24, color='black'))

    fig.update_xaxes(showgrid=False,
                     linecolor='#000',
                     ticks="outside",
                     tickfont=dict(size=18, color='black'),
                     linewidth=1,
                     tickwidth=1,
                     mirror=True)
    fig.update_yaxes(showgrid=False,
                     linecolor='#000',
                     ticks="outside",
                     tickfont=dict(size=18, color='black'),
                     zeroline=False,
                     linewidth=1,
                     tickwidth=1,
                     mirror=True)

    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)",
                      paper_bgcolor='rgba(0,0,0,0)',
                      showlegend=False,
                      title_x=0.5)

    fig.show(renderer="colab")


def plot_bestmatch_results(ts, query, bestmatch_results):
    """
    Visualize the best match results.

    Parameters
    ----------
    ts : numpy.ndarrray
        Time series.

    query : numpy.ndarrray
        Query.

    bestmatch_results : dict 
        The output data found by the best match algorithm.  
    """

    ts_len = ts.shape[0]
    query_len = query.shape[0]

    fig = make_subplots(rows=1, cols=2, column_widths=[0.1, 0.9], subplot_titles=("Query", "Time Series"), horizontal_spacing=0.04)

    # Define Base timeseries (main lines)
    fig.add_trace(go.Scatter(x=np.arange(query_len), y=query, line=dict(color=px.colors.qualitative.Plotly[1])),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=np.arange(ts_len), y=ts, line=dict(color=px.colors.qualitative.Plotly[0])),
                  row=1, col=2)

    for i, start in enumerate(bestmatch_results['index']):
        end = start + query_len
        fig.add_trace(
            go.Scatter(x=np.arange(start, end), y=ts[start:end], line=dict(color=px.colors.qualitative.Plotly[1])),
            row=1, col=2)

    fig.update_annotations(font=dict(size=24, color='black'))

    fig.update_xaxes(showgrid=False,
                     linecolor='#000',
                     ticks="outside",
                     tickfont=dict(size=18, color='black'),
                     linewidth=1,
                     tickwidth=1,
                     mirror=True)
    fig.update_yaxes(showgrid=False,
                     linecolor='#000',
                     ticks="outside",
                     tickfont=dict(size=18, color='black'),
                     zeroline=False,
                     linewidth=1,
                     tickwidth=1,
                     mirror=True)

    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)",
                      paper_bgcolor='rgba(0,0,0,0)',
                      showlegend=False,
                      title_x=0.5)

    fig.show(renderer="colab")



def pie_chart(labels, values, plot_title='Pie chart'):
    """
    Build the pie chart.

    Parameters
    ----------
    labels : numpy.ndarrray
        Sector labels.

    values : numpy.ndarrray
        Values.
    """

    fig = go.Figure(data=[go.Pie(labels=labels, values=values)])

    fig.update_traces(textfont_size=20)
    fig.update_layout(title={'text': plot_title, 'x': 0.5, 'xanchor': 'center'},
                      title_font=dict(size=24, color='black'),
                      legend=dict(font=dict(size=20, color='black')),
                      width=700,
                      height=500
                      )

    fig.show(renderer="colab")