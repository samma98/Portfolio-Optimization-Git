import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import json

from utils.data_processing import fetch_data_from_yahoo, clean_close_data
from utils.optimization import (
    simulate_random_portfolios,
    compute_efficient_frontier,
    compute_max_sharpe_and_min_vol
)
from utils.plotting import plot_full_frontier

# --------------------------------------------------------------------------------
# Initialize Dash App with a Bootstrap "dark" theme
# --------------------------------------------------------------------------------
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
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

# --------------------------------------------------------------------------------
# Navbar
# --------------------------------------------------------------------------------
navbar = dbc.Navbar(
    dbc.Container([
        dbc.NavbarBrand("Portfolio Optimization Tool", className="ms-2", style={"fontWeight": "bold", "align-items": "center", "justify-content": "center", "padding-left": "500px"}),
    ]),
    color="#1e2d3a",
    dark=True,
    
    className="mb-4"
)

# --------------------------------------------------------------------------------
# Left Sidebar Accordion: Data Input & Plot Options
# --------------------------------------------------------------------------------
accordion = dbc.Accordion(
    [
        dbc.AccordionItem([
            
            dbc.Row([
                dbc.Col([
                    dbc.Label("Tickers"),
                    dcc.Dropdown(
                        id="ticker-dropdown",
                        options=TICKER_OPTIONS,
                        value=["AAPL", "GOOG", "MSFT"], 
                        multi=True,
                        placeholder="Select one or more tickers",
                    )
                ], width=12)
            ], className="mb-3"),

            dbc.Row([
                dbc.Col([
                    dbc.Label("Start Date"),
                    dcc.DatePickerSingle(
                        id='start-date',
                        date='2021-01-01',
                        style={"width": "100%"},
                 
                    )
                ], width=6),
                dbc.Col([
                    dbc.Label("End Date"),
                    dcc.DatePickerSingle(
                        id='end-date',
                        style={"width": "100%"},
               
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

            dbc.Button("Fetch & Compute", id="fetch-btn", color="info", n_clicks=0, style={"width": "100%"}),
        ], title="Data Input"),

        dbc.AccordionItem([
            
            dbc.Row([
                dbc.Col([
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
                ], width=12)
            ], className="mb-3"),

            dbc.Row([
                dbc.Col([
                    dbc.Checklist(
                        id='display-options',
                        options=[
                            {"label": "Include Random Portfolios", "value": "RANDOM"}
                        ],
                        value=["RANDOM"],  
                        inline=True,
                        switch=True,  
                        style={"marginTop": "5px"}
                    ),
                ], width=12),
            ], className="mb-3"),
        ], title="Plot Options"),
    ],
    start_collapsed=False,
    flush=True
)

# --------------------------------------------------------------------------------
# Tabs for Graphs
# --------------------------------------------------------------------------------
tabs = dbc.Tabs(
    [
        dbc.Tab(dcc.Graph(id="price-graph", style={"height": "80vh"}), label="Prices Over Time", tab_id="price-tab"),
        dbc.Tab(dcc.Graph(id="corr-graph", style={"height": "80vh"}), label="Correlation Heatmap", tab_id="corr-tab"),
        dbc.Tab(dcc.Graph(id="frontier-graph", style={"height": "80vh"}), label="Efficient Frontier", tab_id="frontier-tab"),
    ],
    id="main-tabs",
    active_tab="price-tab",
    className="mt-2"
)

# --------------------------------------------------------------------------------
# Layout
# --------------------------------------------------------------------------------
app.layout = html.Div([
    navbar,
    dbc.Container([
        dbc.Row([
            dbc.Col([
                
                dbc.Card([
                    dbc.CardHeader("Controls", className="lead"),
                    dbc.CardBody([
                        accordion,
                        dcc.Store(id='stored-price-data'),  
                    ])
                ], className="shadow-sm"),
            ], width=3),

            dbc.Col([
                dcc.Loading(
                    id="loading-graphs",
                    type="circle",
                    children=tabs
                )
            ], width=9)
        ])
    ], fluid=True),

    # Modal for Pie Chart
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("Asset Allocation")),
        dbc.ModalBody([
            dcc.Graph(id="allocation-pie")
        ]),
        dbc.ModalFooter([
            dbc.Button("Close", id="close-modal", className="ms-auto", color="secondary")
        ])
    ], id="allocation-modal", is_open=False)
])

# --------------------------------------------------------------------------------
# Callbacks
# --------------------------------------------------------------------------------

@app.callback(
    Output('stored-price-data', 'data'),
    Input('fetch-btn', 'n_clicks'),
    State('ticker-dropdown', 'value'),
    State('start-date', 'date'),
    State('end-date', 'date')
)
def fetch_and_store_price_data(n_clicks, selected_tickers, start_date, end_date):
    """
    Fetch data from Yahoo Finance for the selected tickers and dates,
    clean it, then store as JSON in dcc.Store.
    """
    if not n_clicks or not selected_tickers:
        return None

    import datetime
    if end_date is None:
        end_date = datetime.datetime.today().strftime('%Y-%m-%d')

    raw_df = fetch_data_from_yahoo(selected_tickers, start_date, end_date)
    close_df = clean_close_data(raw_df)
    if close_df.empty:
        return None

    return close_df.to_json(date_format='iso', orient='split')


@app.callback(
    Output('price-graph', 'figure'),
    Output('corr-graph', 'figure'),
    Output('frontier-graph', 'figure'),
    Input('stored-price-data', 'data'),
    Input('rf-rate', 'value'),
    Input('display-options', 'value'),
    Input('frontier-points', 'value')
)
def update_all_graphs(json_data, rf_rate, display_options, frontier_points):
    """
    Update all three figures: 
      1) Price Over Time, 
      2) Correlation Heatmap, 
      3) Efficient Frontier
    """
    # If no data
    if not json_data:
        no_data_fig = go.Figure().update_layout(
            template="plotly_dark",
            title="No data. Please fetch valid tickers/dates."
        )
        return no_data_fig, no_data_fig, no_data_fig

    if frontier_points is None or frontier_points < 2:
        frontier_points = 50

    price_df = pd.read_json(json_data, orient='split')
    if price_df.empty:
        no_data_fig = go.Figure().update_layout(
            template="plotly_dark",
            title="No valid price data. Check your tickers and date range."
        )
        return no_data_fig, no_data_fig, no_data_fig

    # 1) Price Over Time
    fig_price = px.line(
        price_df,
        x=price_df.index,
        y=price_df.columns,
        title="Prices Over Time"
    )
    fig_price.update_layout(
        template="plotly_dark",
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
        template="plotly_dark",
        margin=dict(l=40, r=40, t=60, b=40),
        coloraxis_showscale=True
    )

    # 3) Efficient Frontier
    frontier_df = compute_efficient_frontier(price_df, points=frontier_points)
    if frontier_df.empty:
        fig_frontier = go.Figure().update_layout(
            template="plotly_dark",
            title="No feasible frontier found for given data."
        )
    else:
        (ret_ms, vol_ms), (ret_mv, vol_mv) = compute_max_sharpe_and_min_vol(price_df, risk_free_rate=rf_rate)
        rand_df = None
        if "RANDOM" in display_options:
            df_temp = simulate_random_portfolios(price_df, risk_free_rate=rf_rate, simulations=2000)
            if not df_temp.empty:
                rand_df = df_temp

        fig_frontier = plot_full_frontier(
            random_df=rand_df,
            frontier_df=frontier_df,
            max_sharpe=(ret_ms, vol_ms),
            min_vol=(ret_mv, vol_mv)
        )
        fig_frontier.update_layout(template="plotly_dark")

    return fig_price, fig_corr, fig_frontier


@app.callback(
    Output("allocation-modal", "is_open"),
    Output("allocation-pie", "figure"),
    Input("frontier-graph", "clickData"),
    Input("close-modal", "n_clicks"),
    State("allocation-modal", "is_open")
)
def show_weights_modal(clickData, close_btn, is_open):
    """
    Show a modal with pie chart if user clicks a point on the Frontier with 'customdata'.
    """
    # Close the modal if user clicks close
    if close_btn and is_open:
        return False, go.Figure()

    if clickData:
        point = clickData["points"][0]
        if "customdata" in point and point["customdata"]:
            weights_str = point["customdata"]
            weights_dict = json.loads(weights_str)

            # Create Pie
            s = pd.Series(weights_dict)
            fig_pie = px.pie(names=s.index, values=s.values, title="Portfolio Weights")
            fig_pie.update_layout(template="plotly_dark")
            return True, fig_pie

    return is_open, go.Figure()


if __name__ == "__main__":
    app.run_server(debug=True)
