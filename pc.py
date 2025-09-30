"""
Pipeline otimizado: associar crimes a ruas reais (OSM), extrair features,
treinar modelos, e gerar mapa de calor interativo (Folium HeatMap).

Saída:
- predicted_heatmaps_heatmap.html  (HTML com filtros e linha do tempo)
"""

import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point

import osmnx as ox

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import folium
from folium.plugins import HeatMap, HeatMapWithTime

# -----------------------------
# 1) Carregar dados simulados
# -----------------------------
csv_path = 'crime_simulated.csv'   # use o dataset maior
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"CSV com dados simulados não encontrado em {csv_path}.")

df = pd.read_csv(csv_path, parse_dates=['datetime'])
crimes_gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs='EPSG:4326')

# -----------------------------
# 2) Baixar malha de ruas
# -----------------------------
place_name = 'Recife, Pernambuco, Brazil'
print('Baixando malha de ruas via OSM (pode demorar alguns segundos)...')
G = ox.graph_from_place(place_name, network_type='drive')
nodes_gdf, edges_gdf = ox.graph_to_gdfs(G)
edges_gdf = edges_gdf.to_crs(epsg=4326)
crimes_gdf = crimes_gdf.to_crs(epsg=4326)

# -----------------------------
# 3) Associar crimes à rua mais próxima
# -----------------------------
print('Associando crimes à rua mais próxima...')
crimes_nearest = gpd.sjoin_nearest(crimes_gdf, edges_gdf.reset_index(drop=True), how='left', distance_col='dist_to_edge')

# Contagem de crimes por rua
counts_by_edge = crimes_nearest.groupby('index_right').size().rename('count').reset_index()

# -----------------------------
# 4) Features por segmento
# -----------------------------
print('Extraindo features por rua...')
edges_gdf = edges_gdf.reset_index(drop=True)
edges_gdf['idx'] = edges_gdf.index
edges_gdf = edges_gdf.merge(counts_by_edge, left_on='idx', right_on='index_right', how='left')
edges_gdf['count'] = edges_gdf['count'].fillna(0)

# Severidade média
sev_by_edge = crimes_nearest.groupby('index_right')['severity'].mean().rename('severity_mean')
edges_gdf['severity_mean'] = edges_gdf['idx'].map(sev_by_edge).fillna(0)

# Proporção por tipo (dummy encoding)
type_pivot = pd.get_dummies(crimes_nearest[['index_right','type']], columns=['type']).groupby('index_right').sum()
edges_gdf = edges_gdf.join(type_pivot, on='idx').fillna(0)

# Variável alvo
edges_gdf['target'] = edges_gdf['count']

# Features
feature_cols = ['severity_mean'] + [col for col in edges_gdf.columns if col.startswith('type_')]
X = edges_gdf[feature_cols].values
y = edges_gdf['target'].values

# -----------------------------
# 5) Treinar modelos
# -----------------------------
print('Treinando modelos...')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

lr = LinearRegression(); lr.fit(X_train, y_train)
rf = RandomForestRegressor(n_estimators=200, random_state=30, n_jobs=-1); rf.fit(X_train, y_train)
svr = SVR(kernel='rbf', C=1.0, epsilon=0.5); svr.fit(X_train_s, y_train)

# Predições
X_all = X
X_all_s = scaler.transform(X_all)
edges_gdf['pred_lr'] = lr.predict(X_all).clip(min=0)
edges_gdf['pred_rf'] = rf.predict(X_all).clip(min=0)
edges_gdf['pred_svr'] = svr.predict(X_all_s).clip(min=0)

# -----------------------------
# 6) Criar HeatMap interativo
# -----------------------------
print('Gerando mapa de calor interativo...')
centro = [crimes_gdf.geometry.y.mean(), crimes_gdf.geometry.x.mean()]
mapa = folium.Map(location=centro, zoom_start=13, tiles="cartodbpositron")

# Função para adicionar HeatMap como camada
def add_heatmap_layer(column, name, color_map=None):
    heat_data = [[geom.y, geom.x, pred] 
                 for geom, pred in zip(edges_gdf.geometry.centroid, edges_gdf[column]) if pred > 0]
    if heat_data:
        heat_layer = HeatMap(
            heat_data,
            radius=12,
            blur=18,
            min_opacity=0.3,
            gradient=color_map,
            name=name
        )
        mapa.add_child(heat_layer)

# 🔹 Modelos preditivos
add_heatmap_layer('pred_rf', 'Random Forest', {0.2:"blue",0.4:"lime",0.6:"orange",1:"red"})
add_heatmap_layer('pred_svr', 'SVM', {0.2:"blue",0.5:"yellow",0.8:"red"})
add_heatmap_layer('pred_lr', 'Linear Regression', {0.3:"cyan",0.6:"purple",1:"red"})

# 🔹 Severidade (camadas de filtro)
for sev in sorted(crimes_gdf['severity'].unique()):
    subset = crimes_gdf[crimes_gdf['severity'] == sev]
    heat_data = subset[['latitude','longitude']].values.tolist()
    if len(heat_data) > 0:
        HeatMap(
            heat_data,
            radius=10,
            blur=20,
            min_opacity=0.3,
            name=f"Severidade {sev}"
        ).add_to(mapa)

# 🔹 Tipos de crime (camadas de filtro)
for crime_type in crimes_gdf['type'].unique():
    subset = crimes_gdf[crimes_gdf['type'] == crime_type]
    heat_data = subset[['latitude','longitude']].values.tolist()
    if len(heat_data) > 0:
        HeatMap(
            heat_data,
            radius=12,
            blur=18,
            min_opacity=0.3,
            name=f"Tipo: {crime_type}"
        ).add_to(mapa)

# 🔹 Linha do tempo (evolução mensal)
df['month'] = df['datetime'].dt.to_period("M")
timeline = []
time_index = []
for month, subset in df.groupby("month"):
    timeline.append(subset[['latitude','longitude']].values.tolist())
    time_index.append(str(month))

HeatMapWithTime(
    timeline,
    index=time_index,
    radius=12,
    auto_play=False,
    max_opacity=0.7,
    name="Evolução temporal"
).add_to(mapa)

# Controle de camadas
folium.LayerControl(collapsed=False).add_to(mapa)

# Exportar
out_html = 'predicted_heatmaps_heatmap.html'
mapa.save(out_html)
print('Mapa gerado:', out_html)
print('Abra o HTML no navegador — filtros + linha do tempo prontos!')
