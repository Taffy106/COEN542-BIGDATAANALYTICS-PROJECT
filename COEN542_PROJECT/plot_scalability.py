import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

df = pd.read_csv("scalability_results.csv")

# ---- Chart 1: execution time & throughput vs data fraction (single core count) ----
# Use the lowest core count as the "baseline" view for this chart.
baseline_cores = df["cores"].min()
df1 = df[df["cores"] == baseline_cores].sort_values("fraction")

fig1 = make_subplots(specs=[[{"secondary_y": True}]])
fig1.add_trace(
    go.Bar(x=df1["fraction"], y=df1["seconds"], name="Execution Time (s)"),
    secondary_y=False,
)
fig1.add_trace(
    go.Scatter(x=df1["fraction"], y=df1["throughput"], name="Throughput (records/s)",
               mode="lines+markers", line=dict(color="orange")),
    secondary_y=True,
)
fig1.update_layout(title=f"Execution Time & Throughput vs. Data Fraction ({baseline_cores} core)")
fig1.update_xaxes(title_text="Data Fraction")
fig1.update_yaxes(title_text="Execution Time (seconds)", secondary_y=False)
fig1.update_yaxes(title_text="Throughput (records/second)", secondary_y=True)
fig1.write_html("scalability_data_fraction.html")

# ---- Chart 2: execution time vs data fraction, one line per core count ----
fig2 = go.Figure()
for cores in sorted(df["cores"].unique()):
    subset = df[df["cores"] == cores].sort_values("fraction")
    fig2.add_trace(go.Scatter(
        x=subset["fraction"], y=subset["seconds"],
        mode="lines+markers", name=f"{cores} core(s)"
    ))
fig2.update_layout(
    title="Execution Time vs. Data Fraction, by Core Count",
    xaxis_title="Data Fraction",
    yaxis_title="Execution Time (seconds)"
)
fig2.write_html("scalability_cores.html")

print("Saved scalability_data_fraction.html and scalability_cores.html")
print("(No PNG export -- avoids the Chrome/Kaleido dependency. See below for a matplotlib option.)")