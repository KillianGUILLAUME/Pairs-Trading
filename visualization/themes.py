"""Professional dark theme for quant dashboards"""
import plotly.graph_objects as go
import plotly.io as pio

# Palette inspirée Bloomberg Terminal + TradingView Pro
BLOOMBERG = {
    "bg":          "#0a0e1a",       # Near-black blue
    "bg_panel":    "#12172b",
    "grid":        "#1f2642",
    "text":        "#d4d9e8",
    "text_dim":    "#7a8299",
    "accent":      "#ff9500",       # Bloomberg orange
    "long":        "#00d4aa",       # Teal green
    "short":       "#ff3b5c",       # Coral red
    "neutral":     "#5b8def",       # Blue
    "warning":     "#ffcc00",
    "purple":      "#a78bfa",
}

QUANT_TEMPLATE = go.layout.Template(
    layout=dict(
        paper_bgcolor=BLOOMBERG["bg"],
        plot_bgcolor=BLOOMBERG["bg_panel"],
        font=dict(family="JetBrains Mono, Menlo, monospace",
                  size=11, color=BLOOMBERG["text"]),
        title=dict(font=dict(size=15, color=BLOOMBERG["accent"])),
        xaxis=dict(gridcolor=BLOOMBERG["grid"], zerolinecolor=BLOOMBERG["grid"],
                   showspikes=True, spikecolor=BLOOMBERG["text_dim"],
                   spikethickness=1, spikedash="dot"),
        yaxis=dict(gridcolor=BLOOMBERG["grid"], zerolinecolor=BLOOMBERG["grid"],
                   showspikes=True, spikecolor=BLOOMBERG["text_dim"],
                   spikethickness=1, spikedash="dot"),
        hoverlabel=dict(bgcolor=BLOOMBERG["bg_panel"],
                        font=dict(family="JetBrains Mono", size=11)),
        legend=dict(bgcolor="rgba(18,23,43,0.8)",
                    bordercolor=BLOOMBERG["grid"], borderwidth=1),
        margin=dict(l=60, r=40, t=60, b=40),
    )
)

pio.templates["quant_dark"] = QUANT_TEMPLATE
pio.templates.default = "quant_dark"
