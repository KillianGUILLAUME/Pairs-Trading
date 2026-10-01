"""Single & dual time series with pro features"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from visualization.themes import BLOOMBERG


def plot_single_series(
    timestamps, prices, symbol: str = "Asset",
    volume=None, log_scale: bool = False,
    show_returns: bool = True,
) -> go.Figure:
    """
    Chart pro : prix + volume + returns distribution.
    Layout : 3 rows (price, volume, returns hist sidebar).
    """
    rows = 2 if volume is not None else 1
    row_heights = [0.75, 0.25] if volume is not None else [1.0]

    fig = make_subplots(
        rows=rows, cols=1, shared_xaxes=True,
        row_heights=row_heights, vertical_spacing=0.03,
        subplot_titles=(f"{symbol}", "Volume") if volume is not None else (f"{symbol}",),
    )

    # Price line avec gradient fill
    fig.add_trace(go.Scatter(
        x=timestamps, y=prices, mode="lines", name=symbol,
        line=dict(color=BLOOMBERG["accent"], width=1.5),
        fill="tozeroy", fillcolor="rgba(255,149,0,0.08)",
        hovertemplate="<b>%{x}</b><br>Price: %{y:.4f}<extra></extra>",
    ), row=1, col=1)

    # Moving averages (context visuel)
    s = pd.Series(prices)
    for window, color in [(50, BLOOMBERG["neutral"]), (200, BLOOMBERG["purple"])]:
        if len(prices) > window:
            ma = s.rolling(window).mean()
            fig.add_trace(go.Scatter(
                x=timestamps, y=ma, mode="lines",
                name=f"MA{window}",
                line=dict(color=color, width=1, dash="dot"),
                opacity=0.7,
            ), row=1, col=1)

    if volume is not None:
        colors = [BLOOMBERG["long"] if prices[i] >= prices[i-1]
                  else BLOOMBERG["short"] for i in range(1, len(prices))]
        colors = [BLOOMBERG["neutral"]] + colors
        fig.add_trace(go.Bar(
            x=timestamps, y=volume, name="Volume",
            marker=dict(color=colors, line=dict(width=0)),
            opacity=0.6,
        ), row=2, col=1)

    if log_scale:
        fig.update_yaxes(type="log", row=1, col=1)

    fig.update_layout(
        title=f"{symbol} — Price Action",
        height=600, showlegend=True,
        hovermode="x unified",
    )
    return fig


def plot_dual_series(
    timestamps, price_a, price_b,
    symbol_a: str, symbol_b: str,
    normalize: bool = True,
    show_correlation: bool = True,
    show_ratio: bool = True,
) -> go.Figure:
    """
    Comparaison deux actifs : prix normalisés + ratio + rolling correlation.
    Layout pro : 3 rows.
    """
    rows = 1 + int(show_ratio) + int(show_correlation)
    titles = [f"{symbol_a} vs {symbol_b}" + (" (Normalized)" if normalize else "")]
    if show_ratio:         titles.append(f"Ratio {symbol_a}/{symbol_b}")
    if show_correlation:   titles.append("Rolling Correlation (60)")

    heights = [0.5] + [0.25] * (rows - 1)

    fig = make_subplots(
        rows=rows, cols=1, shared_xaxes=True,
        row_heights=heights, vertical_spacing=0.04,
        subplot_titles=titles,
    )

    # Normalisation base 100
    pa = np.asarray(price_a, dtype=float)
    pb = np.asarray(price_b, dtype=float)
    if normalize:
        pa_n = pa / pa[0] * 100
        pb_n = pb / pb[0] * 100
    else:
        pa_n, pb_n = pa, pb

    fig.add_trace(go.Scatter(
        x=timestamps, y=pa_n, mode="lines", name=symbol_a,
        line=dict(color=BLOOMBERG["accent"], width=1.5),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=timestamps, y=pb_n, mode="lines", name=symbol_b,
        line=dict(color=BLOOMBERG["neutral"], width=1.5),
    ), row=1, col=1)

    current_row = 2
    if show_ratio:
        ratio = pa / pb
        mean_r, std_r = np.nanmean(ratio), np.nanstd(ratio)
        fig.add_trace(go.Scatter(
            x=timestamps, y=ratio, mode="lines", name="Ratio",
            line=dict(color=BLOOMBERG["purple"], width=1.2),
        ), row=current_row, col=1)
        # Bandes ±2σ
        for mult, label in [(2, "+2σ"), (-2, "-2σ")]:
            fig.add_hline(y=mean_r + mult * std_r, line_dash="dot",
                          line_color=BLOOMBERG["text_dim"],
                          opacity=0.5, row=current_row, col=1)
        fig.add_hline(y=mean_r, line_dash="dash",
                      line_color=BLOOMBERG["warning"], opacity=0.6,
                      row=current_row, col=1)
        current_row += 1

    if show_correlation:
        ret_a = pd.Series(pa).pct_change()
        ret_b = pd.Series(pb).pct_change()
        rolling_corr = ret_a.rolling(60).corr(ret_b)
        colors = np.where(rolling_corr > 0.7, BLOOMBERG["long"],
                 np.where(rolling_corr < 0.3, BLOOMBERG["short"],
                                              BLOOMBERG["neutral"]))
        fig.add_trace(go.Scatter(
            x=timestamps, y=rolling_corr, mode="lines", name="Corr(60)",
            line=dict(color=BLOOMBERG["long"], width=1.2),
            fill="tozeroy", fillcolor="rgba(0,212,170,0.1)",
        ), row=current_row, col=1)
        fig.add_hline(y=0.7, line_dash="dot",
                      line_color=BLOOMBERG["long"], opacity=0.4,
                      row=current_row, col=1)

    fig.update_layout(
        height=200 * rows + 150, hovermode="x unified",
        title=f"<b>{symbol_a}</b> vs <b>{symbol_b}</b> — Pair Analysis",
    )
    return fig
