import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

PROJECT_DIR = Path(r"C:/Users/Chetana/OneDrive/Desktop/SCM/Project")
OUT_DIR     = PROJECT_DIR / "models_and_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(42)

ships = pd.DataFrame({
    "lat": np.random.uniform(-50, 50, 15),
    "lon": np.random.uniform(-150, 150, 15),
    "risk": np.random.rand(15)
})

storms = pd.DataFrame({
    "lat": np.random.uniform(-40, 40, 8),
    "lon": np.random.uniform(-120, 120, 8),
    "risk": np.random.rand(8)
})

ports = pd.DataFrame({
    "lat": np.random.uniform(-30, 30, 6),
    "lon": np.random.uniform(-100, 100, 6),
    "risk": np.zeros(6)
})


fig = go.Figure()

fig.add_trace(go.Scattergeo(
    lon=ships["lon"], lat=ships["lat"],
    text=[f"Ship Risk: {r:.2f}" for r in ships["risk"]],
    mode="markers", name="Ships",
    marker=dict(size=10, color=ships["risk"], colorscale="RdYlGn_r", cmin=0, cmax=1, colorbar=dict(title="Ship Risk"))
))

fig.add_trace(go.Scattergeo(
    lon=storms["lon"], lat=storms["lat"],
    text=[f"Storm Risk: {r:.2f}" for r in storms["risk"]],
    mode="markers", name="Storms",
    marker=dict(size=12, color="blue", symbol="x")
))

fig.add_trace(go.Scattergeo(
    lon=ports["lon"], lat=ports["lat"],
    text="Port", mode="markers", name="Ports",
    marker=dict(size=8, color="black", symbol="square")
))


for i, row in ships.iterrows():
    if row["risk"] > 0.7:  
     
        dists = np.sqrt((ports["lat"] - row["lat"])*2 + (ports["lon"] - row["lon"])*2)
        nearest = ports.iloc[dists.idxmin()]
   
        fig.add_trace(go.Scattergeo(
            lon=[row["lon"], nearest["lon"]],
            lat=[row["lat"], nearest["lat"]],
            mode="lines", name="Reroute Path",
            line=dict(color="red", width=2, dash="dot"),
            opacity=0.7,
            showlegend=(i==0)  
        ))

fig.update_layout(
    title="Global Logistics Risk Network with Rerouting",
    geo=dict(
        scope="world", projection_type="natural earth",
        showland=True, landcolor="lightgray",
        showcountries=True, showocean=True, oceancolor="lightblue"
    ),
    legend=dict(x=0.85, y=0.95)
)

fig.write_html(OUT_DIR / "logistics_risk_rerouting.html", auto_open=True)
print("Saved to:", OUT_DIR / "logistics_risk_rerouting.html")