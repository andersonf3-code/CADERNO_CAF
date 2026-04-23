import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(layout="wide")

@st.cache_data
def carregar():
    df_caf = pd.read_parquet("caf_ativos.parquet")
    df_area = pd.read_parquet("area.parquet")
    df_esp = pd.read_parquet("especificacao.parquet")
    df_renda = pd.read_parquet("renda.parquet")

    df = df_area.merge(df_caf, on="nr_caf", how="inner")

    # renda agregada
    renda = df_renda.groupby("nr_caf")["vl_renda"].sum().reset_index()
    df = df.merge(renda, on="nr_caf", how="left")

    return df

df = carregar()

# =========================
# KPIs
# =========================
st.title("📊 Dashboard CAF")

col1, col2, col3 = st.columns(3)

col1.metric("CAFs ativos", df["nr_caf"].nunique())
col2.metric("Área total", round(df["nr_area"].sum(), 2))
col3.metric("Renda total", round(df["vl_renda"].sum(), 2))

# =========================
# FILTRO
# =========================
ufs = df["sg_uf"].dropna().unique()
uf_sel = st.multiselect("UF", ufs)

if uf_sel:
    df = df[df["sg_uf"].isin(uf_sel)]

# =========================
# GRÁFICO
# =========================
top = df.groupby("sg_uf")["nr_caf"].nunique().reset_index()

fig = px.bar(top, x="sg_uf", y="nr_caf")
st.plotly_chart(fig, use_container_width=True)