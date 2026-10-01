"""Trade path signature clustering and visualization"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from iisignature import sig
from visualization.themes import BLOOMBERG

def plot_trade_clusters(signal, pnl_df: pd.DataFrame, depth: int = 3) -> go.Figure:
    """
    Visualise les trades dans l'espace des signatures.
    Chaque point est un trade, regroupé par sa "forme" géométrique.
    """
    if "position" not in pnl_df or "net_pnl" not in pnl_df:
        return go.Figure().update_layout(title="No position/PnL data for Clustering")

    pos = pnl_df["position"].values
    ts = pd.to_datetime(signal.timestamps, unit="ms")
    spreads = signal.spreads
    zscores = signal.zscores
    
    # 1. Extraction des trades
    trades_data = []
    in_trade = False
    start_idx = 0
    
    for i in range(1, len(pos)):
        if not in_trade and pos[i] != 0:
            in_trade = True
            start_idx = i
        elif in_trade and pos[i] == 0:
            in_trade = False
            # On prend la signature de la trajectoire (spread, zscore) durant le trade
            path_spread = spreads[start_idx:i+1]
            path_z = zscores[start_idx:i+1]
            
            # Normalisation du temps local pour la signature
            t_local = np.linspace(0, 1, len(path_spread))
            path = np.stack([t_local, path_spread, path_z], axis=1).astype(np.float32)
            
            # Calcul de la signature
            s = sig(path, depth)
            
            trades_data.append({
                "signature": s,
                "pnl": pnl_df["net_pnl"].iloc[start_idx:i+1].sum(),
                "duration": (ts[i] - ts[start_idx]).total_seconds() / 3600,
                "type": "Long" if pos[start_idx] > 0 else "Short"
            })

    if len(trades_data) < 3:
        return go.Figure().update_layout(title="Not enough trades for clustering analysis")

    # 2. PCA sur les signatures
    sigs = np.array([t["signature"] for t in trades_data])
    pca = PCA(n_components=2)
    sigs_2d = pca.fit_transform(sigs)
    
    df_plot = pd.DataFrame({
        "x": sigs_2d[:, 0],
        "y": sigs_2d[:, 1],
        "pnl": [t["pnl"] for t in trades_data],
        "duration": [t["duration"] for t in trades_data],
        "type": [t["type"] for t in trades_data]
    })
    
    # 3. Plotting
    fig = go.Figure()
    
    # Color scale : du rouge (perte) au vert (gain)
    fig.add_trace(go.Scatter(
        x=df_plot["x"], y=df_plot["y"],
        mode="markers",
        marker=dict(
            size=12,
            color=df_plot["pnl"],
            colorscale=[[0, BLOOMBERG["short"]], [0.5, BLOOMBERG["bg_panel"]], [1, BLOOMBERG["long"]]],
            showscale=True,
            colorbar=dict(title="Trade PnL"),
            line=dict(width=1, color="white")
        ),
        text=[f"Type: {t}<br>PnL: {p:.2f}<br>Dur: {d:.1f}h" 
              for t, p, d in zip(df_plot["type"], df_plot["pnl"], df_plot["duration"])],
        hovertemplate="%{text}<extra></extra>"
    ))
    
    # Explication des axes PCA
    var_exp = pca.explained_variance_ratio_.sum() * 100
    fig.update_layout(
        title=f"<b>Trade Signature Clusters (PCA 2D - {var_exp:.1f}% Variance)</b>",
        xaxis_title="Path Component 1",
        yaxis_title="Path Component 2",
        template="plotly_dark",
        height=500
    )
    
    return fig
