import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
import json

from utils.data_processing import fetch_data_from_yahoo, clean_close_data
from utils.optimization import (
    simulate_random_portfolios,
    compute_efficient_frontier,
    compute_max_sharpe_and_min_vol
)
from utils.plotting import plot_full_frontier

app = dash.Dash(
    __name__, 
    external_stylesheets=[dbc.themes.BOOTSTRAP],
)
server = app.server

# Common tickers for demonstration
TICKER_OPTIONS = [
    {"label": "Apple (AAPL)", "value": "AAPL"},
    {"label": "Alphabet (GOOG)", "value": "GOOG"},
    {"label": "Microsoft (MSFT)", "value": "MSFT"},
    {"label": "Amazon (AMZN)", "value": "AMZN"},
    {"label": "Tesla (TSLA)", "value": "TSLA"},
    {"label": "IBM (IBM)", "value": "IBM"},
    {"label": "Meta (META)", "value": "META"},
]

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Portfolio Optimization Tool", className="text-center my-4"),
        ], width=12)
    ], justify="center"),

    dbc.Row([
        dbc.Col([
            # Card for data fetching
            dbc.Card([
                dbc.CardBody([
                    html.H5("Fetch Data from Yahoo Finance", className="card-title"),

                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Tickers"),
                            dcc.Dropdown(
                                id="ticker-dropdown",
                                options=TICKER_OPTIONS,
                                value=["AAPL", "GOOG", "MSFT"],  # default selection
                                multi=True,
                                placeholder="Select one or more tickers"
                            )
                        ])
                    ], className="mb-3"),

                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Start Date"),
                            dcc.DatePickerSingle(
                                id='start-date', 
                                date='2021-01-01',
                                style={"width": "100%"}
                            )
                        ], width=6),
                        dbc.Col([
                            dbc.Label("End Date"),
                            dcc.DatePickerSingle(
                                id='end-date',
                                style={"width": "100%"}
                            )
                        ], width=6)
                    ], className="mb-3"),

                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Risk-free Rate"),
                            dbc.Input(
                                id="rf-rate", 
                                type="number", 
                                value=0.03, 
                                step=0.01,
                                style={"width": "100%"}
                            )
                        ], width=6),
                    ], className="mb-3"),

                    dbc.Button("Fetch & Compute", id="fetch-btn", color="primary", n_clicks=0),
                    dcc.Store(id='stored-price-data'),  # to store the cleaned close df
                ])
            ], className="mb-3"),

            # Card for plot options
            dbc.Card([
                dbc.CardBody([
                    html.H5("Plot Options", className="card-title"),
                    dbc.Checklist(
                        id='display-options',
                        options=[
                            {"label": "Include Random Portfolios", "value": "RANDOM"}
                        ],
                        value=["RANDOM"],  # default: show random
                        inline=True
                    ),
                    html.Br(),
                    dbc.Label("Number of Frontier Points:"),
                    dbc.Input(
                        id="frontier-points", 
                        type="number", 
                        min=2, 
                        max=300, 
                        step=1, 
                        value=50,
                        style={"width": "100%"}
                    ),
                ])
            ])
        ], width=3),

        dbc.Col([
            dcc.Loading(
                id="loading-graphs",
                type="dot",
                children=[
                    # Tabs for multiple graphs
                    dbc.Tabs([
                        dbc.Tab(dcc.Graph(id="price-graph", style={"height": "60vh"}), label="Prices Over Time"),
                        dbc.Tab(dcc.Graph(id="corr-graph", style={"height": "60vh"}), label="Correlation Heatmap"),
                        dbc.Tab(dcc.Graph(id="frontier-graph", style={"height": "60vh"}), label="Efficient Frontier"),
                    ], id="main-tabs", active_tab="Prices Over Time")
                ]
            )
        ], width=9)
    ], justify="center"),

    # Modal for Pie Chart
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("Asset Allocation")),
        dbc.ModalBody([
            dcc.Graph(id="allocation-pie")
        ]),
        dbc.ModalFooter([
            dbc.Button("Close", id="close-modal", className="ms-auto")
        ])
    ], id="allocation-modal", is_open=False)
], fluid=True)


@app.callback(
    Output('stored-price-data', 'data'),
    [Input('fetch-btn', 'n_clicks')],
    [
        State('ticker-dropdown', 'value'),
        State('start-date', 'date'),
        State('end-date', 'date')
    ]
)
def fetch_and_store_price_data(n_clicks, selected_tickers, start_date, end_date):
    if n_clicks is None or n_clicks == 0:
        return None
    if not selected_tickers:
        return None

    import datetime
    if end_date is None:
        end_date = datetime.datetime.today().strftime('%Y-%m-%d')

    # 1. Fetch data
    raw_df = fetch_data_from_yahoo(selected_tickers, start_date, end_date)

    # 2. Clean to get close prices only
    close_df = clean_close_data(raw_df)
    if close_df.empty:
        return None

    return close_df.to_json(date_format='iso', orient='split')


@app.callback(
    [
        Output('price-graph', 'figure'),
        Output('corr-graph', 'figure'),
        Output('frontier-graph', 'figure')
    ],
    [
        Input('stored-price-data', 'data'),
        Input('rf-rate', 'value'),
        Input('display-options', 'value'),
        Input('frontier-points', 'value')
    ]
)
def update_all_graphs(json_data, rf_rate, display_options, frontier_points):
    """
    Returns three figures:
      1) Price Over Time
      2) Correlation Heatmap
      3) Efficient Frontier
    """
    # If no data:
    if not json_data:
        empty_fig = go.Figure().update_layout(title="No data. Please fetch valid tickers/dates.")
        return empty_fig, empty_fig, empty_fig

    if frontier_points is None or frontier_points < 2:
        frontier_points = 50

    price_df = pd.read_json(json_data, orient='split')
    if price_df.empty:
        empty_fig = go.Figure().update_layout(title="No valid price data. Check your tickers and date range.")
        return empty_fig, empty_fig, empty_fig

    # 1) Price Over Time figure
    fig_price = px.line(
        price_df,
        x=price_df.index,
        y=price_df.columns,
        title="Prices Over Time"
    )
    fig_price.update_layout(
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
    )

    # 2) Correlation Heatmap
    returns_df = price_df.pct_change().dropna()
    corr = returns_df.corr()
    fig_corr = px.imshow(
        corr, 
        text_auto=True,
        color_continuous_scale="RdBu_r",
        title="Correlation Heatmap"
    )
    fig_corr.update_layout(
        margin=dict(l=50, r=50, t=50, b=50),
        coloraxis_showscale=True
    )

    # 3) Efficient Frontier
    from_frontier = compute_efficient_frontier(price_df, points=frontier_points)
    if from_frontier.empty:
        fig_frontier = go.Figure().update_layout(title="No feasible frontier found for given data.")
    else:
        (ret_ms, vol_ms), (ret_mv, vol_mv) = compute_max_sharpe_and_min_vol(price_df, risk_free_rate=rf_rate)
        rand_df = None
        if "RANDOM" in display_options:
            df_temp = simulate_random_portfolios(price_df, risk_free_rate=rf_rate, simulations=2000)
            if not df_temp.empty:
                rand_df = df_temp
        
        fig_frontier = plot_full_frontier(
            random_df=rand_df,
            frontier_df=from_frontier,
            max_sharpe=(ret_ms, vol_ms),
            min_vol=(ret_mv, vol_mv)
        )

    return fig_price, fig_corr, fig_frontier


@app.callback(
    Output("allocation-modal", "is_open"),
    Output("allocation-pie", "figure"),
    [Input("frontier-graph", "clickData"),
     Input("close-modal", "n_clicks")],
    [State("allocation-modal", "is_open")]
)
def show_weights_modal(clickData, n_close, is_open):
    if n_close and n_close > 0 and is_open:
        return False, go.Figure()

    if clickData:
        point = clickData["points"][0]
        if "customdata" in point:
            weights_str = point["customdata"]
            
            weights_dict = json.loads(weights_str)

            s = pd.Series(weights_dict)
            fig_pie = px.pie(names=s.index, values=s.values, title="Portfolio Weights")
            fig_pie.update_layout(legend=dict(orientation='h', yanchor='bottom', y=-0.2, xanchor='center', x=0.5))
            return True, fig_pie

    return is_open, go.Figure()


if __name__ == '__main__':
    app.run_server(debug=True)
