import plotly.express as px
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import json

def plot_full_frontier(
    random_df: pd.DataFrame, 
    frontier_df: pd.DataFrame,
    max_sharpe: tuple,
    min_vol: tuple,
    include_portfolios: bool = True
):
    """
    random_df (optional): DataFrame with [Returns,Volatility,Sharpe_ratio,Weights].
    frontier_df: DataFrame with ['Returns','Volatility','Weights'] 
                 for the real frontier. The 'Weights' column is a dict of {ticker: weight}.
    max_sharpe, min_vol: (ret, vol) pairs for those special portfolios.
    include_portfolios: If True, we also show the random portfolio scatter.

    The real frontier's 'Weights' are stored as 'customdata' so you can 
    retrieve them on click in a Dash callback to see the actual allocation.
    """

    # 1) Possibly plot random portfolios (blue scatter)
    if include_portfolios and random_df is not None and not random_df.empty:
        fig = px.scatter(
            random_df,
            x="Volatility",
            y="Returns",
            opacity=0.6,
            color_discrete_sequence=["blue"],
            labels={"Volatility": "Volatility (σ)", "Returns": "Return (μ)"},
            title="Markowitz Efficient Frontier (Random + True Frontier)"
        )
    else:
        fig = go.Figure()
        fig.update_layout(
            title="Markowitz Efficient Frontier (No Random Portfolios)",
            xaxis_title="Volatility (σ)",
            yaxis_title="Return (μ)"
        )

    # 2) Frontier line
    frontier_sorted = frontier_df.sort_values("Volatility").reset_index(drop=True)
 
    frontier_sorted["Weights_json"] = frontier_sorted["Weights"].apply(json.dumps)

    fig.add_trace(
        go.Scatter(
            x=frontier_sorted["Volatility"],
            y=frontier_sorted["Returns"],
            mode="lines+markers",
            line=dict(color="orange", width=2),
            name="Efficient Frontier",
            customdata=frontier_sorted["Weights_json"],  # store JSON
            hovertemplate=(
                "Volatility: %{x:.2f}<br>"
                "Return: %{y:.2f}<br>"
                "Weights: %{customdata}<extra></extra>"
            )
        )
    )

    # 3) Max Sharpe point
    ret_ms, vol_ms = max_sharpe
    fig.add_trace(
        go.Scatter(
            x=[vol_ms],
            y=[ret_ms],
            mode='markers',
            marker=dict(color='green', size=10),
            name='Max Sharpe'
        )
    )

    # 4) Min Vol point
    ret_mv, vol_mv = min_vol
    fig.add_trace(
        go.Scatter(
            x=[vol_mv],
            y=[ret_mv],
            mode='markers',
            marker=dict(color='red', size=10),
            name='Min Vol'
        )
    )

    fig.update_layout(
        width=1250,
        height=600,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.01,
            xanchor='center',
            x=0.5
        )
    )
    return fig
