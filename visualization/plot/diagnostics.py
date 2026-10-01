"""Cointegration, half-life, Hurst, stability"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from visualization.themes import BLOOMBERG


def plot_cointegration_stability(stability_df: pd.DataFrame) -> go.Figure:
    """P-value rolling de la cointégration"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=stability_df["end"], y=stability_df["pval"],
        mode="lines+markers", name="Coint p-value",
        line=dict(color=BLOOMBERG["accent"], width=1.5),
        marker=dict(size=4),
    ))
    fig.add_hline(y=0.05, line_dash="dash", line_color=BLOOMBERG["short"],
                  annotation_text="5% threshold")
    fig.add_hrect(y0=0, y1=0.05, fillcolor=BLOOMBERG["long"],
                  opacity=0.1, line_width=0)
    stability = stability_df["is_coint"].mean() * 100
    fig.update_layout(
        title=f"<b>Cointegration Stability — {stability:.1f}% of time significant</b>",
        yaxis_title="p-value", height=400,
    )
    return fig


def plot_pair_scorecard(pair_result) -> go.Figure:
    """Radar/gauge pour évaluer une paire d'un coup d'œil"""
    metrics = {
        "Coint Score":  (1 - (pair_result.eg_pvalue or 1)) * 100,
        "Mean Rev":     max(0, (0.5 - (pair_result.hurst_exponent or 0.5))) * 200,
        "Half-Life":    100 if pair_result.half_life and 10 < pair_result.half_life < 100 else 50,
        "Correlation":  abs(pair_result.correlation or 0) * 100,
    }
    fig = go.Figure(go.Scatterpolar(
        r=list(metrics.values()), theta=list(metrics.keys()),
        fill="toself", fillcolor="rgba(255,149,0,0.2)",
        line=dict(color=BLOOMBERG["accent"], width=2),
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(range=[0, 100], gridcolor=BLOOMBERG["grid"])),
        title=f"<b>{pair_result.symbol_a} × {pair_result.symbol_b}</b>",
        height=400,
    )
    return fig
