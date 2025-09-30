import pandas as pd
from dash import Dash, html, dcc, Output, Input, State
import dash_leaflet as dl
import dash_extensions.javascript as dj
from dash_extensions.enrich import DashProxy
import time

# -----------------------------
# 1) Carregar dataset
# -----------------------------
df = pd.read_csv("crime_simulated.csv", parse_dates=["datetime"])

# -----------------------------
# 2) Inicializar Dash
# -----------------------------
app = Dash(__name__)

# Dropdowns
crime_types = df["type"].unique()
severities = sorted(df["severity"].unique())

app.layout = html.Div([
    html.H1("Mapa de Calor - Criminalidade Recife", style={"textAlign": "center"}),

    html.Div([
        html.Label("Selecione o tipo de crime:"),
        dcc.Dropdown(
            id="crime-dropdown",
            options=[{"label": t, "value": t} for t in crime_types],
            value=list(crime_types),
            multi=True
        ),

        html.Label("Selecione a severidade:"),
        dcc.Dropdown(
            id="severity-dropdown",
            options=[{"label": str(s), "value": s} for s in severities],
            value=severities,
            multi=True
        ),

        html.Label("Linha do tempo (meses):"),
        dcc.Slider(
            id="time-slider",
            min=int(df["datetime"].dt.month.min()),
            max=int(df["datetime"].dt.month.max()),
            value=int(df["datetime"].dt.month.min()),
            marks={int(m): f"Mês {int(m)}" for m in sorted(df["datetime"].dt.month.unique())},
            step=None
        ),

        html.Button("▶️ Play/Pause", id="play-button", n_clicks=0)
    ], style={"width": "60%", "margin": "auto"}),

    html.Div([
        dl.Map(center=[-8.05, -34.9], zoom=12, children=[
            dl.TileLayer(),
            dl.LayerGroup(id="heatmap-layer")
        ], style={"width": "100%", "height": "600px"})
    ])
])

# -----------------------------
# 3) Callback para atualizar mapa
# -----------------------------
@app.callback(
    Output("heatmap-layer", "children"),
    Input("crime-dropdown", "value"),
    Input("severity-dropdown", "value"),
    Input("time-slider", "value")
)
def update_heatmap(selected_types, selected_severity, selected_month):
    # Filtrar dados
    dff = df[
        (df["type"].isin(selected_types)) &
        (df["severity"].isin(selected_severity)) &
        (df["datetime"].dt.month == int(selected_month))
    ]

    if dff.empty:
        return []

    heat_data = dff[["latitude", "longitude"]].values.tolist()

    return [
        dl.Heatmap(positions=heat_data, radius=15, blur=20, max=1.0)
    ]

# -----------------------------
# 4) Callback do Play/Pause
# -----------------------------
@app.callback(
    Output("time-slider", "value"),
    Input("play-button", "n_clicks"),
    State("time-slider", "value"),
    prevent_initial_call=True
)
def animate_timeline(n_clicks, current_value):
    months = sorted(df["datetime"].dt.month.unique())
    months = [int(m) for m in months]
    if current_value not in months:
        return months[0]

    # Quando clica no botão, roda a animação de loop simples
    for m in months:
        time.sleep(0.5)  # velocidade da animação
        yield m  # atualiza o slider a cada passo

# -----------------------------
# 5) Rodar App
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
