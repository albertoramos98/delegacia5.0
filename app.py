# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
from datetime import datetime
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
import sklearn

st.set_page_config(layout="wide", page_title="Delegacia 5.0 — Dashboard Preditivo")

# ------------- Helpers -------------
@st.cache_data
def load_data(ocorr_path="ocorrencias.csv", tipos_path="tipos_crime.csv"):
    df = pd.read_csv(ocorr_path, parse_dates=["data_hora"])
    tipos = pd.read_csv(tipos_path)
    tipos_map = dict(zip(tipos['id_tipo_crime'], tipos['descricao']))
    df['tipo'] = df['tipo_crime_id'].map(tipos_map).fillna("Outro")
    df = df.dropna(subset=['latitude', 'longitude']).reset_index(drop=True)
    df['dia_semana'] = df['data_hora'].dt.day_name()
    df['hora_dia'] = df['data_hora'].dt.hour
    return df


def aggregate_for_model(df):
    # group by bairro, dia_semana, hora_dia, tipo and count events
    agg = df.groupby(['bairro', 'dia_semana', 'hora_dia', 'tipo']).size().reset_index(name='count')
    return agg


def train_poisson(agg):
    # Features: bairro (one-hot), dia_semana (one-hot), hora_dia (numeric), tipo (one-hot)
    X = agg[['bairro', 'dia_semana', 'hora_dia', 'tipo']]
    y = agg['count'].values
    cat_cols = ['bairro', 'dia_semana', 'tipo']

    # Compatibilidade com versões novas e antigas do scikit-learn
    try:
        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)

    pre = ColumnTransformer([
        ('ohe', ohe, cat_cols)
    ], remainder='passthrough')

    model = make_pipeline(pre, PoissonRegressor(alpha=1e-6, max_iter=300))
    model.fit(X, y)
    return model, pre


def make_prediction_grid(df, preproc_and_model):
    # build grid of all combos present in data
    bairros = df['bairro'].unique()
    dias = df['dia_semana'].unique()
    horas = np.arange(0, 24)
    tipos = df['tipo'].unique()

    rows = []
    for b in bairros:
        for d in dias:
            for h in horas:
                for t in tipos:
                    rows.append((b, d, int(h), t))
    grid = pd.DataFrame(rows, columns=['bairro', 'dia_semana', 'hora_dia', 'tipo'])
    model = preproc_and_model
    # predict (model is pipeline)
    preds = model.predict(grid)
    grid['pred_count'] = np.maximum(preds, 0)
    # aggregate per bairro-day-hour (sum across types)
    agg = grid.groupby(['bairro', 'dia_semana', 'hora_dia'])['pred_count'].sum().reset_index()
    # normalize risk 0-1
    agg['risco_norm'] = (agg['pred_count'] - agg['pred_count'].min()) / (agg['pred_count'].max() - agg['pred_count'].min() + 1e-9)
    return grid, agg


# ------------- Load -------------
st.sidebar.title("Configurações")
st.sidebar.markdown("Carregando dados...")
df = load_data()
st.sidebar.success(f"{len(df)} ocorrências carregadas")

# ------------- Train model (cached) -------------
st.sidebar.markdown("Treinando modelo preditivo (Poisson)...")
with st.spinner("Treinando..."):
    agg = aggregate_for_model(df)
    model_pipe, _ = train_poisson(agg)
    grid_types, agg_bairro = make_prediction_grid(df, model_pipe)
st.sidebar.success("Modelo treinado")

# Save previsoes.csv
previsao_out = "previsoes_por_tipo_e_agregadas.csv"
# grid_types: per bairro,dia,hour,tipo ; agg_bairro: per bairro,dia,hour aggregated
grid_types.to_csv("previsoes_por_tipo.csv", index=False)
agg_bairro.to_csv(previsao_out, index=False)

# ------------- UI / Filters -------------
st.title("Delegacia 5.0 — Mapa preditivo e filtros")
col1, col2 = st.columns((1, 3))

with col1:
    st.header("Filtros")
    cidades = sorted(df['cidade'].unique())
    cidade_sel = st.selectbox("Cidade", ["Todas"] + cidades)
    bairros = sorted(df['bairro'].unique())
    bairro_sel = st.selectbox("Bairro", ["Todos"] + bairros)
    dias = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    dia_sel = st.selectbox("Dia da semana", ["Todos"] + dias)
    hora_min, hora_max = st.slider("Faixa de hora", 0, 23, (0, 23))
    tipos = sorted(df['tipo'].unique())
    tipos_sel = st.multiselect("Tipos (multi)", tipos, default=tipos)
    mostrar_prev = st.checkbox("Mostrar previsão futura (risco)", value=True)
    st.markdown("---")
    st.download_button("Download previsões (agregadas)", previsao_out, file_name=previsao_out)

# ------------- Prepare map data -------------
map_df = df.copy()
if cidade_sel != "Todas":
    map_df = map_df[map_df['cidade'] == cidade_sel]
if bairro_sel != "Todos":
    map_df = map_df[map_df['bairro'] == bairro_sel]
if dia_sel != "Todos":
    # convert day names expected by dt.day_name() English; user locale may be PT-BR - we used day_name() English
    map_df = map_df[map_df['dia_semana'] == dia_sel]
map_df = map_df[(map_df['hora_dia'] >= hora_min) & (map_df['hora_dia'] <= hora_max)]
map_df = map_df[map_df['tipo'].isin(tipos_sel)]

# center
if len(map_df) > 0:
    centro = [map_df['latitude'].mean(), map_df['longitude'].mean()]
else:
    centro = [-8.0476, -34.8770]  # Recife center fallback

m = folium.Map(location=centro, zoom_start=12, tiles="CartoDB positron")

# base heat from historical data
heat_data = map_df[['latitude', 'longitude']].values.tolist()
if len(heat_data) > 0:
    HeatMap(heat_data, radius=10, blur=18, name="Histórico").add_to(m)

# add predicted risk layer as circle markers/heat depending on choice
if mostrar_prev:
    # filter aggregated predictions by selected filters
    agg_df = pd.read_csv(previsao_out)
    if cidade_sel != "Todas":
        # no city in predictions, we filter by bairros available in the city from original df
        bairros_city = df[df['cidade'] == cidade_sel]['bairro'].unique().tolist()
        agg_df = agg_df[agg_df['bairro'].isin(bairros_city)]
    if bairro_sel != "Todos":
        agg_df = agg_df[agg_df['bairro'] == bairro_sel]
    if dia_sel != "Todos":
        agg_df = agg_df[agg_df['dia_semana'] == dia_sel]
    agg_df = agg_df[(agg_df['hora_dia'] >= hora_min) & (agg_df['hora_dia'] <= hora_max)]
    # merge centroid coordinates per bairro (from original df)
    centroid = df.groupby('bairro')[['latitude', 'longitude']].mean().reset_index()
    agg_df = agg_df.merge(centroid, on='bairro', how='left')
    # prepare heat points with weight = risco_norm
    heat_pred = agg_df[['latitude', 'longitude', 'risco_norm']].dropna().values.tolist()
    if len(heat_pred) > 0:
        HeatMap(heat_pred, radius=18, blur=30, name="Previsão (Risco)", min_opacity=0.4).add_to(m)

# add layer control
folium.LayerControl(collapsed=False).add_to(m)

with col2:
    st.header("Mapa")
    # show folium map inside streamlit
    st_data = st_folium(m, width=900, height=700)
    st.caption("Use os filtros no painel esquerdo. Dados e previsões geradas a partir de ocorrencias.csv")