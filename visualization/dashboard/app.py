import dash
import plotly.graph_objects as go
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import polars as pl
import numpy as np
import pandas as pd

from visualization.themes import BLOOMBERG
from visualization.plot.timeseries import plot_dual_series
from visualization.plot.spread import plot_spread_diagnostic, plot_rolling_correlation
from visualization.plot.signals import plot_trades_on_spread, plot_anomaly_detection
from visualization.plot.performance import plot_performance_dashboard, plot_pnl_distribution, plot_trade_waterfall
from visualization.plot.optimization import plot_parameter_heatmap
from visualization.plot.clustering import plot_trade_clusters
from research.signals.signal_generator import SignalGenerator
from research.backtest.engine import BacktestEngine, BacktestConfig

# Thème Bootstrap sombre
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],
    suppress_callback_exceptions=True,
)
app.title = "Quant Terminal"

# ─── Layout ───
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H2("⚡ QUANT TERMINAL",
                        style={"color": BLOOMBERG["accent"],
                               "fontFamily": "JetBrains Mono"}), width=8),
        dbc.Col(html.Div(id="live-clock",
                         style={"color": BLOOMBERG["text_dim"],
                                "textAlign": "right"}), width=4),
    ], className="mb-3"),

    dbc.Row([
        dbc.Col([
            dbc.Label("Symbol A"),
            dcc.Dropdown(id="sym-a", value="ADA_USDT",
                         options=[{"label": s, "value": s}
                                  for s in ["ADA_USDT", "AVAX_USDT",
                                            "BTC_USDT", "ETH_USDT"]]),
        ], width=2),
        dbc.Col([
            dbc.Label("Symbol B"),
            dcc.Dropdown(id="sym-b", value="AVAX_USDT",
                         options=[{"label": s, "value": s}
                                  for s in ["ADA_USDT", "AVAX_USDT",
                                            "BTC_USDT", "ETH_USDT"]]),
        ], width=2),
        dbc.Col([
            dbc.Label("Entry Z"),
            dcc.Slider(id="entry-z", min=1.0, max=4.0, step=0.25, value=2.0),
        ], width=2),
        dbc.Col([
            dbc.Label("Exit Z"),
            dcc.Slider(id="exit-z", min=0.0, max=1.5, step=0.1, value=0.3),
        ], width=2),
        dbc.Col([
            dbc.Label("Action"),
            html.Br(),
            dbc.Button("Run Optimization", id="run-opt", color="info", className="w-100"),
        ], width=2),
    ], className="mb-4 g-3"),

    dbc.Tabs([
        dbc.Tab(dcc.Graph(id="pair-chart"),     label="📈 Pair"),
        dbc.Tab(dcc.Graph(id="spread-chart"),   label="🎯 Spread Diagnostic"),
        dbc.Tab(dcc.Graph(id="signals-chart"),  label="🚦 Signals & Correlation"),
        dbc.Tab([
            dcc.Graph(id="perf-chart"),
            dbc.Row([
                dbc.Col(dcc.Graph(id="dist-chart"), width=6),
                dbc.Col(dcc.Graph(id="waterfall-chart"), width=6),
            ])
        ], label="📊 Analytics"),
        dbc.Tab(dcc.Graph(id="anomaly-chart"),  label="🚨 Diagnostics"),
        dbc.Tab(dcc.Graph(id="cluster-chart"),  label="🌀 Signatures"),
        dbc.Tab([
            dbc.Spinner(dcc.Graph(id="opt-chart"), color="info")
        ], label="🧪 Optimization")
    ]),

    dcc.Interval(id="clock", interval=1000),
], fluid=True, style={"backgroundColor": BLOOMBERG["bg"], "minHeight": "100vh"})


# ─── Callbacks ───
@app.callback(Output("live-clock", "children"), Input("clock", "n_intervals"))
def update_clock(_):
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d  %H:%M:%S  UTC")


def _load_pair(sym_a, sym_b):
    data_dir = "data/storage/parquet/1h"
    df_a = pl.read_parquet(f"{data_dir}/{sym_a}.parquet")
    df_b = pl.read_parquet(f"{data_dir}/{sym_b}.parquet")
    df = df_a.join(df_b, on="timestamp", how="inner").sort("timestamp").tail(2000)
    ts = df["timestamp"].to_numpy()
    pa = df["close"].to_numpy()
    pb = df["close_right"].to_numpy()
    return ts, pa, pb


@app.callback(
    [Output("pair-chart", "figure"),
     Output("spread-chart", "figure"),
     Output("signals-chart", "figure"),
     Output("perf-chart", "figure"),
     Output("dist-chart", "figure"),
     Output("waterfall-chart", "figure"),
     Output("anomaly-chart", "figure"),
     Output("cluster-chart", "figure")],
    [Input("sym-a", "value"), Input("sym-b", "value"),
     Input("entry-z", "value"), Input("exit-z", "value")],
)
def update_charts(sym_a, sym_b, entry_z, exit_z):
    ts, pa, pb = _load_pair(sym_a, sym_b)
    sig_gen = SignalGenerator(entry_threshold=entry_z, exit_threshold=exit_z)
    signal = sig_gen.generate(ts, pa, pb, symbol_a=sym_a, symbol_b=sym_b)

    bt_engine = BacktestEngine(BacktestConfig())
    bt_res = bt_engine.run(signal)
    pnl_df = bt_res['df']
    pnl_df.index = pd.to_datetime(ts, unit='ms')
    
    # Triple-row subplot: Spread, Z-score, and Correlation
    from plotly.subplots import make_subplots
    fig_sig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, 
        row_heights=[0.45, 0.35, 0.2], 
        vertical_spacing=0.04,
        subplot_titles=("Spread + Trades", "Z-score + Signals", "Rolling Correlation")
    )
    
    # On délègue à plot_trades_on_spread mais on va ré-affecter ses traces aux bonnes lignes
    # On pourrait aussi refaire la logique ici pour plus de précision
    f1 = plot_trades_on_spread(signal, positions=bt_res['positions'])
    f2 = plot_rolling_correlation(signal)
    
    # Traces de f1 : Spread est en haut, Z-score au milieu
    # f1.data[0] est le spread, f1.data[1] est le z-score (selon plot_trades_on_spread)
    # Les markers suivent.
    
    for tr in f1.data:
        # Si la trace appartient à l'axe Y original du z-score (y2), on la met ligne 2
        # Sinon ligne 1
        row_idx = 1 if tr.yaxis == "y" else 2
        fig_sig.add_trace(tr, row=row_idx, col=1)
        
    for tr in f2.data:
        fig_sig.add_trace(tr, row=3, col=1)
        
    fig_sig.update_layout(height=850, title="<b>Market Signals & Context</b>", template="plotly_dark")

    return (
        plot_dual_series(ts, pa, pb, sym_a, sym_b),
        plot_spread_diagnostic(signal, entry_threshold=entry_z, exit_threshold=exit_z),
        fig_sig,
        plot_performance_dashboard(pnl_df),
        plot_pnl_distribution(pnl_df['net_pnl'] / pnl_df['capital'].shift(1).fillna(100000)),
        plot_trade_waterfall(pnl_df),
        plot_anomaly_detection(signal),
        plot_trade_clusters(signal, pnl_df),
    )


@app.callback(
    Output("opt-chart", "figure"),
    Input("run-opt", "n_clicks"),
    [State("sym-a", "value"), State("sym-b", "value")]
)
def run_optimization(n_clicks, sym_a, sym_b):
    if n_clicks is None: return go.Figure().update_layout(title="Click 'Run Optimization' to start")
    
    ts, pa, pb = _load_pair(sym_a, sym_b)
    entries = np.linspace(1.5, 3.5, 8)
    exits = np.linspace(0.1, 1.0, 8)
    
    results = []
    engine = BacktestEngine(BacktestConfig())
    sig_gen = SignalGenerator()
    
    for en in entries:
        for ex in exits:
            sig_gen.entry_threshold = en
            sig_gen.exit_threshold = ex
            signal = sig_gen.generate(ts, pa, pb, symbol_a=sym_a, symbol_b=sym_b)
            res = engine.run(signal)
            results.append({
                "entry_threshold": en,
                "exit_threshold": ex,
                "sharpe": res["metrics"]["sharpe"]
            })
            
    df_opt = pd.DataFrame(results)
    return plot_parameter_heatmap(df_opt)


if __name__ == "__main__":
    app.run(debug=True, port=8050)
