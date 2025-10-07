# folium_app_interativo.py
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline

# ---------------------------
# Paths
# ---------------------------
OCORR = "ocorrencias.csv"
TIPOS = "tipos_crime.csv"

# ---------------------------
# Load Data
# ---------------------------
df = pd.read_csv(OCORR, parse_dates=['data_hora'])
tipos = pd.read_csv(TIPOS)
tipos_map = dict(zip(tipos['id_tipo_crime'], tipos['descricao']))
df['tipo'] = df['tipo_crime_id'].map(tipos_map).fillna('Outro')
df['dia_semana'] = df['data_hora'].dt.day_name()
df['hora_dia'] = df['data_hora'].dt.hour

# ---------------------------
# Aggregate for Model
# ---------------------------
agg = df.groupby(['bairro','dia_semana','hora_dia','tipo']).size().reset_index(name='count')
X = agg[['bairro','dia_semana','hora_dia','tipo']]
y = agg['count'].values
cat_cols = ['bairro','dia_semana','tipo']

# Handle OneHotEncoder compatibility
try:
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
except TypeError:
    ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)

pre = ColumnTransformer([('ohe', ohe, cat_cols)], remainder='passthrough')
model = make_pipeline(pre, PoissonRegressor(alpha=1e-6, max_iter=300))
model.fit(X, y)

# ---------------------------
# Build Prediction Grid
# ---------------------------
bairros = df['bairro'].unique()
dias = df['dia_semana'].unique()
horas = np.arange(0,24)
tipos_unique = df['tipo'].unique()

rows = [(b,d,int(h),t) for b in bairros for d in dias for h in horas for t in tipos_unique]
grid = pd.DataFrame(rows, columns=['bairro','dia_semana','hora_dia','tipo'])
grid['pred'] = model.predict(grid)

# Aggregate predictions per bairro/day/hour
pred_ag = grid.groupby(['bairro','dia_semana','hora_dia'])['pred'].sum().reset_index()
pred_ag['risco'] = (pred_ag['pred'] - pred_ag['pred'].min())/(pred_ag['pred'].max()-pred_ag['pred'].min()+1e-9)

# ---------------------------
# Save predictions
# ---------------------------
grid.to_csv('previsoes_por_tipo.csv', index=False)
pred_ag.to_csv('previsoes_agregadas.csv', index=False)

# ---------------------------
# Build Map (centered in Recife)
# ---------------------------
center = [df['latitude'].mean(), df['longitude'].mean()]
m = folium.Map(location=center, zoom_start=12, tiles='CartoDB dark_matter')

# Historical Heat
hist_heat = df[['latitude','longitude']].values.tolist()
HeatMap(hist_heat, radius=10, blur=18, name='Histórico').add_to(m)

# Compute centroids per bairro
centroids = df.groupby('bairro')[['latitude','longitude']].mean().reset_index()

# ---------------------------
# Add interactive JS filter (day & hour)
# ---------------------------
# Prepare data in JS-friendly format
pred_js_data = pred_ag.merge(centroids, on='bairro', how='left').dropna(subset=['latitude','longitude'])
pred_js_data['dia_semana'] = pred_js_data['dia_semana'].astype(str)
pred_js_data['hora_dia'] = pred_js_data['hora_dia'].astype(int)
pred_js_data['risco'] = pred_js_data['risco'].round(3)

# Convert to JSON-like list
records = pred_js_data.to_dict(orient='records')

# Sidebar + dropdowns + JS for filtering
sidebar = f"""
<div id="sidebar" style="position: fixed; top: 10px; left: 10px; z-index:9999; width:280px;
background:white; padding:12px; border-radius:8px; box-shadow: 0 2px 6px rgba(0,0,0,0.3);">
<h4 style="margin:0 0 8px 0">Filtros Interativos</h4>
<p style="margin:0 0 8px 0; font-size:13px">Escolha dia e hora para visualizar previsão de risco</p>

<label for="dia">Dia da semana:</label>
<select id="dia" style="width:100%; margin-bottom:8px;">
  {''.join([f'<option value="{d}">{d}</option>' for d in dias])}
</select>

<label for="hora">Hora do dia:</label>
<select id="hora" style="width:100%; margin-bottom:8px;">
  {''.join([f'<option value="{h}">{h}</option>' for h in horas])}
</select>

<button onclick="updateHeat()" style="width:100%; padding:6px; margin-top:4px;">Atualizar Mapa</button>
</div>

<script>
let map = window.map || {{}};  // Folium map reference
let heatLayer;

const data = {records};

function updateHeat(){{
    const dia = document.getElementById('dia').value;
    const hora = parseInt(document.getElementById('hora').value);

    const filtered = data.filter(d=>d.dia_semana===dia && d.hora_dia===hora);

    const heatPoints = filtered.map(d => [d.latitude, d.longitude, d.risco]);

    if(heatLayer){{
        map.removeLayer(heatLayer);
    }}

    heatLayer = L.heatLayer(heatPoints, {{radius:18, blur:28}}).addTo(map);
}}
</script>
"""

m.get_root().html.add_child(folium.Element(sidebar))

# Add Layer Control
folium.LayerControl(collapsed=False).add_to(m)

# ---------------------------
# Save map
# ---------------------------
out_html = "predicted_heatmaps_interativo.html"
m.save(out_html)
print("Mapa interativo gerado:", out_html)
print("Previsões salvas: previsoes_por_tipo.csv, previsoes_agregadas.csv")
