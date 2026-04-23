import pandas as pd

CHUNK = 200_000

# =========================
# 1. DADOS CAF (filtra ativos)
# =========================
ativos = []

for chunk in pd.read_csv("01_DADOS_CAF.csv", sep=";", chunksize=CHUNK):
    chunk["nr_caf"] = chunk["nr_caf"].astype(str)
    
    chunk["st_ativo"] = chunk["st_ativo"].astype(str).str.upper()
    chunk = chunk[chunk["st_ativo"].isin(["SIM", "TRUE", "1", "ATIVO"])]
    
    ativos.append(chunk[["nr_caf"]])

df_caf = pd.concat(ativos).drop_duplicates()
df_caf.to_parquet("caf_ativos.parquet")

cafs = set(df_caf["nr_caf"])

# =========================
# 2. AREA
# =========================
areas = []

for chunk in pd.read_csv("05_AREA.csv", sep=";", chunksize=CHUNK):
    chunk["nr_caf"] = chunk["nr_caf"].astype(str)
    chunk = chunk[chunk["nr_caf"].isin(cafs)]
    areas.append(chunk)

pd.concat(areas).to_parquet("area.parquet")

# =========================
# 3. ESPECIFICAÇÃO
# =========================
esp = []

for chunk in pd.read_csv("03_ESPECIFICACAO.csv", sep=";", chunksize=CHUNK):
    chunk["nr_caf"] = chunk["nr_caf"].astype(str)
    chunk = chunk[chunk["nr_caf"].isin(cafs)]
    esp.append(chunk)

pd.concat(esp).to_parquet("especificacao.parquet")

# =========================
# 4. RENDA
# =========================
renda = []

for chunk in pd.read_csv("07_PRODUCAO.csv", sep=";", chunksize=CHUNK):
    chunk["nr_caf"] = chunk["nr_caf"].astype(str)
    chunk = chunk[chunk["nr_caf"].isin(cafs)]
    renda.append(chunk)

pd.concat(renda).to_parquet("renda.parquet")

print("✅ Pré-processamento concluído!")