# gera_delegacia_db.py
import csv
import random
from datetime import datetime, timedelta
import os

random.seed(42)

out_dir = "delegacia_5_0_csv"
os.makedirs(out_dir, exist_ok=True)

tipos_crime = [
    "Furto", "Roubo", "Homicídio", "Tráfico de drogas",
    "Violência doméstica", "Estupro", "Latrocínio", "Vandalismo",
    "Lesão corporal", "Sequestro"
]

armas = ["Não utilizada", "Arma de fogo", "Arma branca", "Outros"]
condicoes_climaticas = ["Sol", "Nublado", "Chuva leve", "Chuva forte", "Tempestade"]

events = [
    {"nome_evento": "Carnaval de Rua", "tipo_evento": "Público", "localizacao": "Centro", "data_inicio": "2025-02-15 18:00:00", "data_fim": "2025-02-18 04:00:00"},
    {"nome_evento": "Show no Marco Zero", "tipo_evento": "Público", "localizacao": "Marco Zero, Recife", "data_inicio": "2025-07-10 20:00:00", "data_fim": "2025-07-10 23:30:00"},
    {"nome_evento": "Festa Junina", "tipo_evento": "Público", "localizacao": "Olinda", "data_inicio": "2025-06-23 17:00:00", "data_fim": "2025-06-24 02:00:00"},
    {"nome_evento": "Jockey Club Evento", "tipo_evento": "Privado", "localizacao": "Várzea", "data_inicio": "2025-05-05 19:00:00", "data_fim": "2025-05-05 23:00:00"}
]

cities = {
    "Recife": [
        ("Boa Vista", -8.0591, -34.8847, "50010-000"),
        ("Casa Amarela", -8.0221, -34.9186, "52011-000"),
        ("Ibura", -8.1135, -34.9323, "51110-000"),
        ("Boa Viagem", -8.1265, -34.9011, "51021-000"),
        ("Várzea", -8.0434, -34.9519, "50110-000"),
        ("Santo Amaro", -8.0488, -34.8853, "50020-000"),
    ],
    "Olinda": [
        ("Carmo", -8.0125, -34.8542, "53110-000"),
        ("Rio Doce", -7.9849, -34.8368, "53116-000"),
        ("Casa Caiada", -7.9887, -34.8417, "53114-000"),
        ("Varadouro", -8.0145, -34.8559, "53110-200"),
    ],
    "Jaboatão dos Guararapes": [
        ("Candeias", -8.2039, -34.9221, "54430-000"),
        ("Piedade", -8.1776, -34.9154, "54420-000"),
        ("Prazeres", -8.1863, -34.9231, "54310-000"),
        ("Curado", -8.0962, -34.9704, "54320-000"),
    ],
    "Paulista": [
        ("Maranguape I", -7.9403, -34.8683, "53400-000"),
        ("Janga", -7.9367, -34.8359, "53410-000"),
        ("Aurora", -7.9395, -34.8720, "53420-000"),
        ("Paratibe", -7.9329, -34.8766, "53430-000"),
    ]
}

# renda media
renda_media = {}
for city, bairros in cities.items():
    for bairro, lat, lon, cep in bairros:
        renda_media[bairro] = round(random.uniform(1200, 8000), 2)

def write_list_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

write_list_csv(os.path.join(out_dir, "tipos_crime.csv"), ["id_tipo_crime", "descricao"],
               [[i+1, t] for i, t in enumerate(tipos_crime)])
write_list_csv(os.path.join(out_dir, "armas.csv"), ["id_arma", "descricao"],
               [[i+1, a] for i, a in enumerate(armas)])
write_list_csv(os.path.join(out_dir, "condicoes_climaticas.csv"), ["id_condicao", "descricao"],
               [[i+1, c] for i, c in enumerate(condicoes_climaticas)])
write_list_csv(os.path.join(out_dir, "eventos.csv"),
               ["id_evento", "nome_evento", "tipo_evento", "localizacao", "data_inicio", "data_fim"],
               [[i+1, e["nome_evento"], e["tipo_evento"], e["localizacao"], e["data_inicio"], e["data_fim"]]
                for i, e in enumerate(events)])

N = 50000
start_date = datetime.now() - timedelta(days=365)
with open(os.path.join(out_dir, "ocorrencias.csv"), "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    header = ["id_ocorrencia","data_hora","dia_semana","hora_dia","cep_rua","bairro","cidade",
              "latitude","longitude","tipo_crime_id","arma_utilizada_id","status_investigacao",
              "condicoes_climaticas_id","evento_local_id","renda_media_regiao","presenca_cameras_seguranca"]
    writer.writerow(header)
    for i in range(1, N+1):
        city = random.choices(list(cities.keys()), weights=[0.45, 0.15, 0.25, 0.15], k=1)[0]
        bairros = cities[city]
        base_weights = [4 if b[0] in ["Ibura","Prazeres","Boa Vista","Piedade","Varadouro"] else 2 for b in bairros]
        bairro_idx = random.choices(range(len(bairros)), weights=base_weights, k=1)[0]
        bairro, base_lat, base_lon, cep = bairros[bairro_idx]
        lat = base_lat + random.uniform(-0.007, 0.007)
        lon = base_lon + random.uniform(-0.007, 0.007)
        delta_days = random.randint(0, 365)
        delta_seconds = random.randint(0, 86400-1)
        dt = start_date + timedelta(days=delta_days, seconds=delta_seconds)
        dia_semana = dt.strftime("%A")
        hora_dia = dt.hour
        tipo_id = random.randint(1, len(tipos_crime))
        arma_id = random.choices([1,2,3,4], weights=[0.5,0.3,0.15,0.05], k=1)[0]
        cond_id = random.randint(1, len(condicoes_climaticas))
        evento_id = ""
        for idx, ev in enumerate(events, start=1):
            ev_start = datetime.strptime(ev["data_inicio"], "%Y-%m-%d %H:%M:%S")
            ev_end = datetime.strptime(ev["data_fim"], "%Y-%m-%d %H:%M:%S")
            if ev_start <= dt <= ev_end and random.random() < 0.4:
                evento_id = idx
                break
        status = random.choices(["Aberto","Concluído"], weights=[0.7, 0.3], k=1)[0]
        renda = renda_media[bairro]
        cameras = random.choices([1,0], weights=[0.6, 0.4], k=1)[0]
        writer.writerow([i, dt.strftime("%Y-%m-%d %H:%M:%S"), dia_semana, hora_dia, cep, bairro, city,
                         round(lat,8), round(lon,8), tipo_id, arma_id, status, cond_id,
                         evento_id, renda, cameras])

print("CSV files generated in", out_dir)
