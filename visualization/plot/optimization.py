"""Optimization and sensitivity analysis plots"""
import numpy as np
import plotly.graph_objects as go
from visualization.themes import BLOOMBERG

def plot_parameter_heatmap(
    results_df, 
    x_param: str = "entry_threshold", 
    y_param: str = "exit_threshold",
    metric: str = "sharpe"
) -> go.Figure:
    """
    Génère une heatmap de performance en fonction de deux paramètres.
    results_df doit avoir les colonnes x_param, y_param et metric.
    """
    pivot = results_df.pivot(index=y_param, columns=x_param, values=metric)
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=pivot.index,
        colorscale=[[0, BLOOMBERG["short"]], [0.5, BLOOMBERG["bg_panel"]], [1, BLOOMBERG["long"]]],
        colorbar=dict(title=metric.upper()),
        hovertemplate=f"{x_param}: %{{x}}<br>{y_param}: %{{y}}<br>{metric}: %{{z:.4f}}<extra></extra>"
    ))
    
    fig.update_layout(
        title=f"<b>Sensitivity Analysis: {metric.upper()}</b>",
        xaxis_title=x_param.replace("_", " ").title(),
        yaxis_title=y_param.replace("_", " ").title(),
        height=500,
        template="plotly_dark"
    )
    
    return fig
