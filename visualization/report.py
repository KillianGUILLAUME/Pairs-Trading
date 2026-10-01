"""One-shot HTML report generation"""
from pathlib import Path
import plotly.io as pio

def export_report(figures: dict, output: str = "report.html", title: str = "Quant Report"):
    """Concatène plusieurs figures en un seul HTML autonome"""
    html_parts = [f"""
    <html><head><title>{title}</title>
    <style>
      body {{ background: #0a0e1a; color: #d4d9e8;
              font-family: 'JetBrains Mono', monospace; margin: 0; padding: 20px; }}
      h1 {{ color: #ff9500; }}
      .section {{ margin: 30px 0; padding: 20px;
                   background: #12172b; border-radius: 8px; }}
    </style></head><body>
    <h1>⚡ {title}</h1>
    """]
    for name, fig in figures.items():
        html_parts.append(f'<div class="section"><h2>{name}</h2>')
        html_parts.append(pio.to_html(fig, include_plotlyjs="cdn", full_html=False))
        html_parts.append("</div>")
    html_parts.append("</body></html>")
    Path(output).write_text("\n".join(html_parts))
    print(f"✅ Report saved → {output}")
