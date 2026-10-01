"""Equity curve, drawdown, underwater"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from visualization.themes import BLOOMBERG


def plot_performance_dashboard(pnl_df: pd.DataFrame, benchmark=None) -> go.Figure:
    """
    Dashboard complet de performance :
      - Equity curve (log option)
      - Drawdown underwater
      - Rolling Sharpe
      - Monthly returns heatmap (bonus)
    """
    if "equity" in pnl_df:
        equity = pnl_df["equity"]
    elif "capital" in pnl_df:
        equity = pnl_df["capital"]
    else:
        equity = pnl_df["cum_pnl"]
    
    returns = equity.pct_change().fillna(0)
    
    # Drawdown
    running_max = equity.cummax()
    dd = (equity - running_max) / running_max * 100
    
    # Rolling Sharpe (annualisé, window=24*30)
    rolling_sharpe = (returns.rolling(24*30).mean() /
                      returns.rolling(24*30).std()) * np.sqrt(24*365)

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.5, 0.25, 0.25], vertical_spacing=0.05,
        subplot_titles=("Equity Curve", "Drawdown (%)", "Rolling Sharpe (30d)"),
    )

    # Equity
    fig.add_trace(go.Scatter(
        x=equity.index, y=equity.values, name="Strategy",
        line=dict(color=BLOOMBERG["accent"], width=1.8),
        fill="tozeroy", fillcolor="rgba(255,149,0,0.05)",
    ), row=1, col=1)

    if benchmark is not None:
        fig.add_trace(go.Scatter(
            x=benchmark.index, y=benchmark.values, name="Benchmark",
            line=dict(color=BLOOMBERG["text_dim"], width=1, dash="dot"),
        ), row=1, col=1)

    # Drawdown underwater
    fig.add_trace(go.Scatter(
        x=dd.index, y=dd.values, name="Drawdown",
        line=dict(color=BLOOMBERG["short"], width=1),
        fill="tozeroy", fillcolor="rgba(255,59,92,0.3)",
    ), row=2, col=1)

    # Rolling Sharpe
    colors = ["rgba(0,212,170,0.3)" if s > 0 else "rgba(255,59,92,0.3)"
              for s in rolling_sharpe.fillna(0)]
    fig.add_trace(go.Bar(
        x=rolling_sharpe.index, y=rolling_sharpe.values,
        name="Sharpe", marker_color=colors, marker_line_width=0,
    ), row=3, col=1)
    fig.add_hline(y=1.0, line_dash="dash",
                  line_color=BLOOMBERG["long"], opacity=0.4, row=3, col=1)

    fig.update_layout(
        height=800, title="<b>Strategy Performance Dashboard</b>",
        hovermode="x unified", showlegend=True,
    )
    return fig


def plot_monthly_heatmap(returns: pd.Series) -> go.Figure:
    """Heatmap des returns mensuels — classique chez les hedge funds"""
    monthly = returns.resample("ME").apply(lambda x: (1 + x).prod() - 1) * 100
    df = monthly.to_frame("ret")
    df["year"]  = df.index.year
    df["month"] = df.index.month
    pivot = df.pivot(index="year", columns="month", values="ret")

    fig = go.Figure(go.Heatmap(
        z=pivot.values, x=[f"{m:02d}" for m in pivot.columns],
        y=pivot.index,
        colorscale=[[0, BLOOMBERG["short"]], [0.5, BLOOMBERG["bg_panel"]],
                    [1, BLOOMBERG["long"]]],
        zmid=0,
        text=pivot.round(2).values, texttemplate="%{text}%",
        textfont=dict(size=10),
        hovertemplate="Year: %{y}<br>Month: %{x}<br>Return: %{z:.2f}%<extra></extra>",
    ))
    fig.update_layout(title="<b>Monthly Returns (%)</b>", height=400)
    return fig


def plot_pnl_distribution(returns: pd.Series) -> go.Figure:
    """Histogramme des rendements avec VaR et CVaR"""
    rets = returns[returns != 0]
    if len(rets) == 0:
        return go.Figure().update_layout(title="No data for PnL Distribution")

    var_95 = np.percentile(rets, 5)
    cvar_95 = rets[rets <= var_95].mean()

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=rets, nbinsx=50, name="Returns",
        marker_color=BLOOMBERG["accent"], opacity=0.75,
        hovertemplate="Return: %{x:.2%}<br>Count: %{y}<extra></extra>"
    ))

    # Lignes VaR/CVaR
    fig.add_vline(x=var_95, line_dash="dash", line_color=BLOOMBERG["warning"],
                  annotation_text=f"VaR 95%: {var_95:.2%}")
    fig.add_vline(x=cvar_95, line_dash="dot", line_color=BLOOMBERG["short"],
                  annotation_text=f"CVaR 95%: {cvar_95:.2%}")

    fig.update_layout(
        title="<b>Return Distribution & Tail Risk</b>",
        xaxis_title="Step Return", yaxis_title="Frequency",
        height=400, showlegend=False
    )
    return fig


def plot_trade_waterfall(pnl_df: pd.DataFrame) -> go.Figure:
    """Visualise les cycles de vie des trades individuels"""
    if "position" not in pnl_df:
        return go.Figure().update_layout(title="No position data for Waterfall")

    pos = pnl_df["position"].values
    net_pnl = pnl_df["net_pnl"].values
    ts = pnl_df.index

    # Extraction des trades
    trades = []
    in_trade = False
    start_idx = 0
    
    for i in range(1, len(pos)):
        if not in_trade and pos[i] != 0:
            in_trade = True
            start_idx = i
        elif in_trade and pos[i] == 0:
            in_trade = False
            trade_pnl = net_pnl[start_idx:i+1].sum()
            trades.append({
                "start": ts[start_idx],
                "end": ts[i],
                "pnl": trade_pnl,
                "duration": (ts[i] - ts[start_idx])
            })
    
    if not trades:
        return go.Figure().update_layout(title="No completed trades found")

    df_trades = pd.DataFrame(trades)
    df_trades["color"] = df_trades["pnl"].apply(
        lambda x: BLOOMBERG["long"] if x > 0 else BLOOMBERG["short"]
    )

    fig = go.Figure()
    # On affiche chaque trade comme une barre horizontale (Gantt-like)
    # L'axe Y représente l'ordre chronologique ou le PnL
    for i, row in df_trades.iterrows():
        fig.add_trace(go.Scatter(
            x=[row["start"], row["end"]],
            y=[row["pnl"], row["pnl"]],
            mode="lines+markers",
            name=f"Trade {i}",
            line=dict(color=row["color"], width=4),
            marker=dict(size=8),
            hovertemplate=(f"PnL: {row['pnl']:.2f}<br>"
                           f"Duration: {row['duration']}<extra></extra>")
        ))

    fig.update_layout(
        title="<b>Trade Lifecycle & Impact</b>",
        xaxis_title="Time", yaxis_title="Trade Net PnL",
        height=500, showlegend=False,
        hovermode="closest"
    )
    return fig
