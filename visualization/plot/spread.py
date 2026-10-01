"""The money chart: spread, Kalman, Z-score, regimes"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from visualization.themes import BLOOMBERG


def plot_spread_diagnostic(
    signal,                        # PairSignal
    entry_threshold: float = 2.0,
    exit_threshold: float = 0.3,
    show_volatility: bool = True,
) -> go.Figure:
    """
    LE chart central : tout ce qui se passe sur le spread.
    
    Rows:
      1. Prices (A vs B, normalisés)
      2. Spread + Bollinger Bands
      3. Z-score avec bandes d'entry/exit colorées
      4. Hedge ratio dynamique (beta Kalman)
      5. Realized volatility (rolling)
    """
    ts       = signal.timestamps
    pa, pb   = signal.price_a, signal.price_b
    spreads  = signal.spreads
    betas    = signal.betas
    zscores  = signal.zscores

    rows = 4 + int(show_volatility)
    heights = [0.22, 0.22, 0.22, 0.18] + ([0.16] if show_volatility else [])
    titles = [
        f"{signal.symbol_a} vs {signal.symbol_b} (Normalized)",
        "Spread (log-price residual) + BB(100, 2σ)",
        f"Z-score — Entry ±{entry_threshold} / Exit ±{exit_threshold}",
        "Kalman β (dynamic hedge ratio)",
    ]
    if show_volatility:
        titles.append("Realized Volatility (60)")

    fig = make_subplots(
        rows=rows, cols=1, shared_xaxes=True,
        row_heights=heights, vertical_spacing=0.025,
        subplot_titles=titles,
    )

    # ── Row 1: Prices normalisés ──
    pa_n = pa / pa[0] * 100
    pb_n = pb / pb[0] * 100
    fig.add_trace(go.Scatter(x=ts, y=pa_n, name=signal.symbol_a,
        line=dict(color=BLOOMBERG["accent"], width=1.3)), row=1, col=1)
    fig.add_trace(go.Scatter(x=ts, y=pb_n, name=signal.symbol_b,
        line=dict(color=BLOOMBERG["neutral"], width=1.3)), row=1, col=1)

    # Bollinger Bands (Rolling 100)
    spread_series = pd.Series(spreads)
    spread_mean = spread_series.mean()  # Global mean for the center line
    rolling_mean = spread_series.rolling(100).mean()
    rolling_std  = spread_series.rolling(100).std()
    upper_band = rolling_mean + 2 * rolling_std
    lower_band = rolling_mean - 2 * rolling_std

    fig.add_trace(go.Scatter(
        x=ts, y=spreads, name="Spread",
        line=dict(color=BLOOMBERG["purple"], width=1.2),
        fill="tozeroy",
        fillcolor="rgba(167,139,250,0.08)",
    ), row=2, col=1)

    # Bollinger Bands
    fig.add_trace(go.Scatter(
        x=ts, y=upper_band, name="Upper BB",
        line=dict(color=BLOOMBERG["short"], width=1, dash="dot"),
        opacity=0.5), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=ts, y=lower_band, name="Lower BB",
        line=dict(color=BLOOMBERG["long"], width=1, dash="dot"),
        opacity=0.5), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=ts, y=rolling_mean, name="Rolling Mean",
        line=dict(color=BLOOMBERG["warning"], width=0.8, dash="dash"),
        opacity=0.4), row=2, col=1)

    fig.add_hline(y=spread_mean, line_dash="dash",
                  line_color=BLOOMBERG["warning"], opacity=0.5, row=2, col=1)

    # ── Row 3: Z-score ──
    fig.add_trace(go.Scatter(
        x=ts, y=zscores, name="Z-score",
        line=dict(color=BLOOMBERG["text"], width=1.2),
    ), row=3, col=1)

    # Zones colorées Z-score
    fig.add_hrect(y0=entry_threshold, y1=max(np.nanmax(zscores), entry_threshold+1),
                  fillcolor=BLOOMBERG["short"], opacity=0.1, line_width=0, row=3, col=1)
    fig.add_hrect(y0=min(np.nanmin(zscores), -entry_threshold-1), y1=-entry_threshold,
                  fillcolor=BLOOMBERG["long"], opacity=0.1, line_width=0, row=3, col=1)

    for y, color, dash in [
        (entry_threshold,  BLOOMBERG["short"],  "dash"),
        (-entry_threshold, BLOOMBERG["long"],   "dash"),
        (exit_threshold,   BLOOMBERG["warning"], "dot"),
        (-exit_threshold,  BLOOMBERG["warning"], "dot"),
        (0,                BLOOMBERG["text_dim"], "solid"),
    ]:
        fig.add_hline(y=y, line_dash=dash, line_color=color, opacity=0.6, row=3, col=1)

    # ── Row 4: Beta Kalman ──
    fig.add_trace(go.Scatter(
        x=ts, y=betas, name="β (Kalman)",
        line=dict(color=BLOOMBERG["long"], width=1.2),
    ), row=4, col=1)

    # ── Row 5: Volatilité ──
    if show_volatility:
        vol = spread_series.diff().rolling(60).std() * np.sqrt(24)
        fig.add_trace(go.Scatter(
            x=ts, y=vol, name="σ(60)",
            line=dict(color=BLOOMBERG["warning"], width=1),
            fill="tozeroy", fillcolor="rgba(255,204,0,0.1)",
        ), row=5, col=1)

    fig.update_layout(
        height=200 * rows + 100,
        title=f"<b>Spread Diagnostic — {signal.symbol_a} × {signal.symbol_b}</b>",
        hovermode="x unified", showlegend=True,
    )
    return fig


def plot_rolling_correlation(signal, window: int = 30 * 24) -> go.Figure:
    """Affiche la corrélation glissante entre l'Asset A et l'Asset B"""
    df = pd.DataFrame({
        "a": signal.price_a,
        "b": signal.price_b
    }, index=pd.to_datetime(signal.timestamps, unit="ms"))
    
    corr = df["a"].rolling(window).corr(df["b"])
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=corr.index, y=corr.values, name="Rolling Corr",
        line=dict(color=BLOOMBERG["accent"], width=1.5),
        fill="tozeroy", fillcolor="rgba(0,188,212,0.1)"
    ))
    
    fig.add_hline(y=0.9, line_dash="dash", line_color=BLOOMBERG["long"], opacity=0.5)
    fig.add_hline(y=0.5, line_dash="dot", line_color=BLOOMBERG["warning"], opacity=0.5)
    
    fig.update_layout(
        title=f"<b>Rolling Correlation ({window}h)</b>",
        yaxis=dict(range=[0, 1]), height=300,
        yaxis_title="Pearson Correlation"
    )
    return fig
