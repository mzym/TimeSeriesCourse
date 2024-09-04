import numpy as np
import pandas as pd
import math

# for visualization
import plotly
from plotly.subplots import make_subplots
from plotly.offline import init_notebook_mode
import plotly.graph_objs as go
import plotly.express as px
plotly.offline.init_notebook_mode(connected=True)


def plot_ts(ts: np.ndarrray, title: str = 'Input Time Series') -> None:
    """
    Plot the time series

    Parameters
    ----------
    ts: time series
    title: title of plot
    """

    n = ts.shape[0]

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=np.arange(n), y=ts, line=dict(width=3)))

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



def plot_motifs(mp: dict, top_k_motifs: dict) -> None:
    """
    Plot the top-k motifs in time series and matrix profile

    Parameters
    ----------
    mp: the matrix profile structure
    top_k_motifs: top-k motifs
    """

    top_k = len(top_k_motifs['indices'])
    n = len(mp['data']['ts1'])
    m = mp['m']

    num_cols = 2
    num_rows = 2 + math.ceil(top_k / num_cols)

    titles = ['Time Series with top-k motifs', 'Matrix Profile'] + [f"Top-{i+1} motifs" for i in range(top_k)]

    fig = make_subplots(rows=num_rows, cols=num_cols,
                        specs=[[{"colspan": 2}, None]]*2 + [[{}, {}]]*(num_rows-2),
                        shared_xaxes=False,
                        vertical_spacing=0.1,
                        subplot_titles=titles)

    fig.add_trace(go.Scatter(x=np.arange(n), y=mp['data']['ts1'], line=dict(color='grey'), name="Time Series"), row=1, col=1)

    for i in range(top_k):
        left_motif_idx = top_k_motifs['indices'][i][0]
        right_motif_idx = top_k_motifs['indices'][i][1]
        x = np.arange(left_motif_idx, right_motif_idx+m)
        num_values_between_motif = right_motif_idx - (left_motif_idx+m)
        y = np.concatenate((mp['data']['ts1'][left_motif_idx:left_motif_idx+m], np.full([1, num_values_between_motif], np.nan)[0], mp['data']['ts1'][right_motif_idx:right_motif_idx+m]))
        color_i = i % len(px.colors.qualitative.Plotly)
        fig.add_trace(go.Scatter(x=x, y=y, name=f"Top-{i+1} motifs", line=dict(color=px.colors.qualitative.Plotly[color_i])), row=1, col=1) #line=dict(color=px.colors.qualitative.Plotly[i+1])

    fig.add_trace(go.Scatter(x=np.arange(n), y=mp['mp'], line=dict(color='grey', width=2), name="Matrix Profile"), row=2, col=1)

    for i in range(top_k):
        motifs_mp = [mp['mp'][motif_idx] for motif_idx in top_k_motifs['indices'][i]]
        motifs_idx = list(top_k_motifs['indices'][i])
        color_i = i % len(px.colors.qualitative.Plotly)
        fig.add_trace(go.Scatter(x=motifs_idx, y=motifs_mp, mode='markers', marker=dict(symbol='star', color=px.colors.qualitative.Plotly[color_i], size=15), name=f"Top-{i+1} motifs"), row=2, col=1) # color='red',

    for i in range(top_k):
        col = int(i % num_cols) + 1
        row = 2 + int(i / num_cols) + 1
        left_motif_idx = top_k_motifs['indices'][i][0]
        right_motif_idx = top_k_motifs['indices'][i][1]
        color_i = i % len(px.colors.qualitative.Plotly)
        fig.add_trace(go.Scatter(x=np.arange(m), y=mp['data']['ts1'][left_motif_idx:left_motif_idx+m], line=dict(color=px.colors.qualitative.Plotly[color_i]), showlegend = False), row=row, col=col)
        fig.add_trace(go.Scatter(x=np.arange(m), y=mp['data']['ts1'][right_motif_idx:right_motif_idx+m], line=dict(color=px.colors.qualitative.Plotly[color_i]), showlegend = False), row=row, col=col)


    fig.update_annotations(font=dict(size=22, color='black'))

    fig.update_xaxes(showgrid=False,
                     title_font=dict(size=22, color='black'),
                     linecolor='#000',
                     ticks='outside',
                     tickfont=dict(size=18, color='black'),
                     linewidth=2,
                     tickwidth=2)
    fig.update_yaxes(showgrid=False,
                     title_font=dict(size=22, color='black'),
                     linecolor='#000',
                     ticks='outside',
                     tickfont=dict(size=18), color='black',
                     zeroline=False,
                     linewidth=2,
                     tickwidth=2)

    fig.update_layout(title_font=dict(size=24, color='black'),
                      plot_bgcolor='rgba(0,0,0,0)',
                      paper_bgcolor='rgba(0,0,0,0)', 
                      height=1300)

    fig.show(renderer="colab")


def plot_discords(mp: dict, top_k_discords: dict) -> None:
    """
    Plot the top-k discords in time series and matrix profile

    Parameters
    ----------
    mp: matrix profile structure
    top_k_discords: top-k discords
    """

    top_k = len(top_k_discords['indices'])
    n = len(mp['data']['ts1'])
    m = mp['m']

    fig = make_subplots(rows=2, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.02)

    # plot time series with discords
    fig.add_trace(go.Scatter(x=np.arange(n), y=mp['data']['ts1'], line=dict(color='#636EFA'), name="Time Series"), row=1, col=1)

    for i in range(top_k):
        discord_idx = top_k_discords['indices'][i]
        fig.add_trace(go.Scatter(x=np.arange(discord_idx, discord_idx+m), y=mp['data']['ts1'][discord_idx:discord_idx+m], line=dict(color='red'), name=f"Top-{i+1} discord"), row=1, col=1)

    # plot mp
    discords_idx = top_k_discords['indices']
    discords_mp = [mp['mp'][discord_idx] for discord_idx in discords_idx]

    fig.add_trace(go.Scatter(x=np.arange(n), y=mp['mp'], line=dict(color='#636EFA', width=2), name="Matrix Profile"), row=2, col=1)
    fig.add_trace(go.Scatter(x=discords_idx, y=discords_mp, mode='markers', marker=dict(symbol='star', color='red', size=7), name="Discords"), row=2, col=1)

    fig.update_layout(title_text="Top-k discords in time series")

    fig.update_xaxes(showgrid=False,
                     title_font=dict(size=22, color='black'),
                     linecolor='#000',
                     ticks="outside",
                     tickfont=dict(size=18, color='black'),
                     linewidth=2,
                     tickwidth=2)
    fig.update_yaxes(showgrid=False,
                     title_font=dict(size=22, color='black'),
                     linecolor='#000',
                     ticks="outside",
                     tickfont=dict(size=18), color='black',
                     zeroline=False,
                     linewidth=2,
                     tickwidth=2)

    fig.update_layout(title_font=dict(size=24, color='black'),
                      plot_bgcolor="rgba(0,0,0,0)",
                      paper_bgcolor='rgba(0,0,0,0)')

    fig.show(renderer="colab")


def plot_segmentation(mp: dict, threshold: float) -> None:
    """
    Plot the segmented time series
    
    Parameters
    ----------
    mp: the matrix profile structure
    threshold: threshold
    """

    n = len(mp['data']['ts1'])

    fig = make_subplots(rows=2, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.15,
                        subplot_titles=("The segmented time series", "Matrix Profile"))

    fig.add_trace(go.Scatter(x=np.arange(n), y=mp['data']['ts1'], line=dict(color='#636EFA'), name="Time Series", showlegend = False), row=1, col=1)
    fig.add_trace(go.Scatter(x=np.arange(n), y=mp['mp'], line=dict(color='#636EFA', width=2), name="Matrix Profile", showlegend = False), row=2, col=1)

    fig.add_hline(y=threshold, line_width=3, line_dash="dash", line_color="red", row=2, col=1)

    fig.update_annotations(font=dict(size=22, color='black'))
    fig.update_xaxes(showgrid=False,
                     title_font=dict(size=22, color='black'),
                     linecolor='#000',
                     ticks="outside",
                     tickfont=dict(size=18, color='black'),
                     linewidth=2,
                     tickwidth=2)
    fig.update_yaxes(showgrid=False,
                     title_font=dict(size=22, color='black'),
                     linecolor='#000',
                     ticks="outside",
                     tickfont=dict(size=18), color='black',
                     zeroline=False,
                     linewidth=2,
                     tickwidth=2)

    fig.update_layout(title_font=dict(size=24, color='black'),
                      plot_bgcolor="rgba(0,0,0,0)",
                      paper_bgcolor='rgba(0,0,0,0)', height=700)

    fig.show(renderer="colab")
