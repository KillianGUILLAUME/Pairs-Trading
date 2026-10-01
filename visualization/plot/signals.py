"""Visualize where you actually trade"""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from visualization.themes import BLOOMBERG


def plot_trades_on_spread(signal, positions=None, pnl_df=None) -> go.Figure:
    """
    Spread + Z-score avec markers d'entrée/sortie.
    Si pnl_df fourni, colore les trades par PnL.
    """
    ts = signal.timestamps
    spreads = signal.spreads
    zscores = signal.zscores

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.6, 0.4], vertical_spacing=0.04,
        subplot_titles=("Spread + Trades", "Z-score + Signals"),
    )

    fig.add_trace(go.Scatter(x=ts, y=spreads, name="Spread",
        line=dict(color=BLOOMBERG["purple"], width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=ts, y=zscores, name="Z-score",
        line=dict(color=BLOOMBERG["text"], width=1)), row=2, col=1)

    # Entries
    long_idx  = np.where(signal.entry_long)[0]
    short_idx = np.where(signal.entry_short)[0]
    
    # Exits: si on a les positions, on ne montre que les sorties RÉELLES
    if positions is not None:
        # Une sortie est une transition de non-zéro vers zéro
        pos_series = np.asarray(positions)
        real_exits = np.where((pos_series[:-1] != 0) & (pos_series[1:] == 0))[0] + 1
        exit_idx = real_exits
    else:
        # Sinon on montre le signal brut (mais c'est souvent bruité)
        exit_idx  = np.where(signal.exit_signal)[0]

    def _add_markers(idx, name, color, symbol, row):
        if len(idx) == 0: return
        fig.add_trace(go.Scatter(
            x=ts[idx], y=(spreads if row == 1 else zscores)[idx],
            mode="markers", name=name,
            marker=dict(color=color, size=10, symbol=symbol,
                        line=dict(width=1, color="white")),
            hovertemplate=f"<b>{name}</b><br>%{{x}}<br>Value: %{{y:.4f}}<extra></extra>",
        ), row=row, col=1)

    for row in [1, 2]:
        _add_markers(long_idx,  "Long Spread",  BLOOMBERG["long"],  "triangle-up",   row)
        _add_markers(short_idx, "Short Spread", BLOOMBERG["short"], "triangle-down", row)
        _add_markers(exit_idx,  "Exit",         BLOOMBERG["warning"], "x",           row)

    fig.update_layout(
        height=700, title="<b>Trading Signals</b>",
        hovermode="x unified",
    )
    return fig


def plot_anomaly_detection(signal) -> go.Figure:
    """
    Visualisation de l'Oeil de Sauron : les résidus de Kalman (Innovations).
    Détecte quand le spread se comporte de manière "anormale" par rapport au modèle.
    """
    ts = signal.timestamps
    # On utilise les kalman_zscores (innovations normalisées)
    k_z = signal.kalman_zscores
    
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.3, 0.7], vertical_spacing=0.05,
        subplot_titles=("Spread Context", "Kalman Prediction Error (Standardized)")
    )
    
    # Row 1: Context
    fig.add_trace(go.Scatter(
        x=ts, y=signal.spreads, name="Spread",
        line=dict(color=BLOOMBERG["purple"], width=1)
    ), row=1, col=1)
    
    # Row 2: Kalman Z-scores
    fig.add_trace(go.Scatter(
        x=ts, y=k_z, name="Kalman Z",
        line=dict(color=BLOOMBERG["accent"], width=1.2),
        fill="tozeroy", fillcolor="rgba(255,214,0,0.1)"
    ), row=2, col=1)
    
    # Zones d'anomalie (|Z| > 3)
    # On utilise des barres verticales rouges pour les anomalies extrêmes
    anomalies_idx = np.where(np.abs(k_z) > 4.0)[0]
    if len(anomalies_idx) > 0:
        fig.add_trace(go.Scatter(
            x=ts[anomalies_idx], y=k_z[anomalies_idx],
            mode="markers", name="Model Divergence",
            marker=dict(color=BLOOMBERG["short"], size=8, symbol="star-diamond"),
            hovertemplate="<b>Anomaly!</b><br>Z-Kalman: %{y:.2f}<extra></extra>"
        ), row=2, col=1)

    # Danger lines
    for y in [3.0, -3.0]:
        fig.add_hline(y=y, line_dash="dash", line_color=BLOOMBERG["short"], opacity=0.5, row=2, col=1)
        
    fig.update_layout(
        height=500, title="<b>Anomaly Detection: 'Eye of Sauron' (Kalman Residuals)</b>",
        hovermode="x unified", showlegend=True,
        template="plotly_dark"
    )
    
    return fig
