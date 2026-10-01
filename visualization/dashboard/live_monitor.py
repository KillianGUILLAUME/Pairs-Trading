import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import json
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

# Import themes
from visualization.themes import BLOOMBERG

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
app.title = "Institutional Live Monitor"

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STATE_PATH = os.path.join(project_root, "live_platform", "state.json")

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("🛰️ INSTITUTIONAL LIVE MONITOR", style={"color": BLOOMBERG["accent"], "fontFamily": "JetBrains Mono"}), width=8),
        dbc.Col(html.Div(id="live-clock", style={"textAlign": "right", "fontSize": "20px", "color": BLOOMBERG["text_dim"]}), width=4),
    ], className="mt-4 mb-4"),

    # KPI Row
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H4("Scan Health", className="card-title"),
                html.H2(id="kpi-scan", style={"color": "#00ff00"})
            ])
        ], color="dark", outline=True), width=3),
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H4("Valid Signals", className="card-title"),
                html.H2(id="kpi-signals", style={"color": "#ffaa00"})
            ])
        ], color="dark", outline=True), width=3),
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H4("Veto Count", className="card-title"),
                html.H2(id="kpi-veto", style={"color": "#ff4444"})
            ])
        ], color="dark", outline=True), width=3),
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H4("Portfolio Exposure", className="card-title"),
                html.H2(id="kpi-exposure", style={"color": "#00ddee"})
            ])
        ], color="dark", outline=True), width=3),
    ], className="mb-4"),

    # Main Grid
    dbc.Row([
        dbc.Col([
            html.H3("🌍 Universal Pair Matrix (153 Pairs)", className="mb-3"),
            dcc.Graph(id="signal-matrix")
        ], width=12)
    ], className="mb-4"),

    # Bottom Logs
    dbc.Row([
        dbc.Col([
            html.H3("📜 Execution Log", className="mb-2"),
            html.Div(id="execution-log", style={
                "backgroundColor": "#111", 
                "padding": "15px", 
                "border": f"1px solid {BLOOMBERG['grid']}",
                "fontFamily": "Courier New",
                "height": "200px",
                "overflowY": "scroll"
            })
        ], width=12)
    ]),

    dcc.Interval(id="monitor-update", interval=5000), # 5s refresh
], fluid=True, style={"backgroundColor": "#050505", "minHeight": "100vh", "color": "white"})

@app.callback(
    [Output("kpi-scan", "children"),
     Output("kpi-signals", "children"),
     Output("kpi-veto", "children"),
     Output("kpi-exposure", "children"),
     Output("signal-matrix", "figure"),
     Output("execution-log", "children"),
     Output("live-clock", "children")],
    Input("monitor-update", "n_intervals")
)
def update_monitor(_):
    now_str = datetime.now().strftime("%H:%M:%S")
    
    if not os.path.exists(STATE_PATH):
        return "N/A", "N/A", "N/A", "N/A", go.Figure(), "Waiting for state...", now_str

    with open(STATE_PATH, "r") as f:
        state = json.load(f)

    # KPIs
    scan_txt = f"{state['pairs_scanned']} Pairs"
    sig_txt = f"{state['valid_signals']}"
    veto_txt = f"{state['veto_count']}"
    exp_txt = f"{sum(state['allocations'].values()):.2%}"

    # Matrix Visualization
    probs = state["probs"]
    df_m = pd.DataFrame(list(probs.items()), columns=["pair", "prob"])
    # Split pair names for better grid display if needed, but here we just do a bar or scatter matrix
    
    fig_matrix = go.Figure()
    fig_matrix.add_trace(go.Bar(
        x=df_m["pair"],
        y=df_m["prob"],
        marker_color=np.where(df_m["prob"] >= 0.60, "#00ff00", "#ff4444"),
        name="Oracle Confidence"
    ))
    fig_matrix.add_hline(y=0.60, line_dash="dash", line_color="white", annotation_text="Veto Threshold")
    fig_matrix.update_layout(
        template="plotly_dark",
        height=400,
        margin=dict(l=20, r=20, t=20, b=100),
        xaxis={'tickangle': 45, 'title': ''},
        yaxis={'title': 'Oracle Prob'}
    )

    # Logs
    logs = [html.P(f"> {l}", style={"margin": "2px"}) for l in state["execution_log"]]

    return scan_txt, sig_txt, veto_txt, exp_txt, fig_matrix, logs, now_str

if __name__ == "__main__":
    app.run(debug=True, port=8051)
