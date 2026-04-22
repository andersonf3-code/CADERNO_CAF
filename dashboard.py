import math
import io
import unicodedata
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.parse import urlparse
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# =========================================================
# CONFIGURAÇÃO DA PÁGINA
# =========================================================
st.set_page_config(
    page_title="Projeto Cadernos do CAF - BI Premium",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# PARÂMETROS
# =========================================================
ARQUIVO_DADOS_CAF_PADRAO = "https://drive.usercontent.google.com/download?id=1MCxERa2nUi7C4GwqP6uKUq9dkXf_qMWR&confirm=t"
ARQUIVO_ESPECIFICACAO_PADRAO = "https://drive.usercontent.google.com/download?id=1iTdx_lmzKk7CwSqeD2TGuPWW0BtgkLF_&confirm=t"
ARQUIVO_AREA_PADRAO = "https://drive.usercontent.google.com/download?id=1g_d7mo_AVWTADSpApylaxC8PAOgP9m1J&confirm=t"
ARQUIVO_RENDA_PADRAO = "https://drive.usercontent.google.com/download?id=1JIqUWwORIhz56TIgEBVIqlCK1ogmp8KN&confirm=t"

CARACTERIZACAO_AREA_ALVO = "DEMAIS AGRICULTORES FAMILIARES"

MAX_PONTOS_MAPA = 7000
MAX_LINHAS_TABELA_POR_PAGINA = 1000
MAX_EXPORT_LINHAS = 500000
MAX_REGISTROS_SEM_FILTRO = 200000

# =========================================================
# ESTILO
# =========================================================
st.markdown("""
<style>
.block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
}
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
    border: 1px solid rgba(15,23,42,0.08);
    padding: 14px 16px;
    border-radius: 18px;
    box-shadow: 0 8px 24px rgba(15,23,42,0.06);
}
div[data-testid="stMetricLabel"] {
    color: #475569 !important;
    font-weight: 700;
}
div[data-testid="stMetricValue"] {
    color: #0f172a !important;
}
.kpi-card {
    background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
    border: 1px solid rgba(15,23,42,0.08);
    border-radius: 18px;
    padding: 16px 18px;
    box-shadow: 0 8px 24px rgba(15,23,42,0.06);
}
.kpi-label {
    color: #475569;
    font-size: 0.95rem;
    font-weight: 700;
    margin-bottom: 8px;
}
.kpi-value {
    color: #0f172a;
    font-size: 2rem;
    font-weight: 800;
    line-height: 1.15;
}
.kpi-sub {
    color: #64748b;
    font-size: 0.82rem;
    margin-top: 6px;
}
.small-note {
    color: #64748b;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# FUNÇÕES UTILITÁRIAS
# =========================================================
def remover_acentos(texto):
    if pd.isna(texto):
        return texto
    texto = str(texto)
    return ''.join(
        c for c in unicodedata.normalize('NFKD', texto)
        if not unicodedata.combining(c)
    )


def padronizar_nome_coluna(col):
    col = str(col).strip()
    col = remover_acentos(col)
    col = col.lower()
    for ch in [" ", "-", "/", ".", "(", ")"]:
        col = col.replace(ch, "_")
    while "__" in col:
        col = col.replace("__", "_")
    return col.strip("_")


def padronizar_colunas(df):
    df = df.copy()
    df.columns = [padronizar_nome_coluna(c) for c in df.columns]
    return df


def normalizar_nr_caf(serie):
    return (
        serie.astype(str)
        .str.strip()
        .str.replace("\u00a0", "", regex=False)
        .str.replace(r"\.0$", "", regex=True)
    )


def formatar_inteiro(valor):
    if pd.isna(valor):
        return "0"
    return f"{int(valor):,}".replace(",", ".")


def formatar_decimal_br(valor, casas=2):
    if pd.isna(valor):
        valor = 0
    return f"{valor:,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def contar_cafs_unicos(df):
    if df is None or len(df) == 0 or "nr_caf" not in df.columns:
        return 0
    return df["nr_caf"].astype(str).nunique()


def eh_url(valor):
    if not isinstance(valor, str):
        return False
    parsed = urlparse(valor)
    return parsed.scheme in ("http", "https") and bool(parsed.netloc)

def baixar_bytes_url(url):
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req) as resp:
        return resp.read()

def detectar_separador(caminho_ou_bytes):
    if eh_url(caminho_ou_bytes):
        conteudo = baixar_bytes_url(caminho_ou_bytes)[:5000]
        amostra = conteudo.decode("utf-8", errors="ignore")
    elif isinstance(caminho_ou_bytes, (str, Path)):
        amostra = Path(caminho_ou_bytes).read_text(encoding="utf-8", errors="ignore")[:5000]
    else:
        pos = caminho_ou_bytes.tell()
        conteudo = caminho_ou_bytes.read(5000)
        caminho_ou_bytes.seek(pos)
        try:
            amostra = conteudo.decode("utf-8", errors="ignore")
        except Exception:
            amostra = str(conteudo)

    return ";" if amostra.count(";") > amostra.count(",") else ","

def ler_csv_seguro(origem):
    sep = detectar_separador(origem)

    if eh_url(origem):
        conteudo = baixar_bytes_url(origem)
        buffer = io.BytesIO(conteudo)
        try:
            return pd.read_csv(buffer, sep=sep, encoding="utf-8", low_memory=False)
        except UnicodeDecodeError:
            buffer.seek(0)
            return pd.read_csv(buffer, sep=sep, encoding="latin1", low_memory=False)

    if isinstance(origem, (str, Path)):
        try:
            return pd.read_csv(origem, sep=sep, encoding="utf-8", low_memory=False)
        except UnicodeDecodeError:
            return pd.read_csv(origem, sep=sep, encoding="latin1", low_memory=False)

    try:
        origem.seek(0)
        return pd.read_csv(origem, sep=sep, encoding="utf-8", low_memory=False)
    except UnicodeDecodeError:
        origem.seek(0)
        return pd.read_csv(origem, sep=sep, encoding="latin1", low_memory=False)


def normalizar_texto_serie(serie):
    return (
        serie.astype(str)
        .str.strip()
        .apply(remover_acentos)
        .str.upper()
    )


def normalizar_texto_valor(valor):
    if pd.isna(valor):
        return ""
    return remover_acentos(str(valor).strip()).upper()


def converter_bool_sim_nao(serie):
    s = serie.astype(str).str.strip().apply(remover_acentos).str.upper()
    mapa = {
        "TRUE": "SIM",
        "FALSE": "NAO",
        "1": "SIM",
        "0": "NAO",
        "SIM": "SIM",
        "NAO": "NAO",
        "S": "SIM",
        "N": "NAO",
        "ATIVO": "SIM",
        "ATIVA": "SIM",
        "INATIVO": "NAO",
        "INATIVA": "NAO",
        "YES": "SIM",
        "NO": "NAO"
    }
    return s.map(mapa).fillna(s)


def garantir_coluna(df, alternativas, nome_final):
    for alt in alternativas:
        if alt in df.columns:
            return df.rename(columns={alt: nome_final})
    return df


def encontrar_coluna(df, alternativas, obrigatoria=False, nome_logico="coluna"):
    for alt in alternativas:
        if alt in df.columns:
            return alt
    if obrigatoria:
        raise ValueError(f"Coluna obrigatória ausente: {nome_logico}. Alternativas aceitas: {alternativas}")
    return None


def converter_categorias(df):
    df = df.copy()
    for col in [
        "sg_uf",
        "nm_municipio",
        "ds_tipo_area",
        "ds_condicao_dominio",
        "ds_tipo_localizacao_area",
        "st_imovel_principal",
        "ds_produto",
        "nm_caracterizacao_area"
    ]:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df


def criar_card_kpi(label, value, subtexto=""):
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-sub">{subtexto}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def tema_plotly_claro():
    return dict(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#0f172a"),
        margin=dict(l=20, r=20, t=55, b=20),
    )


def para_excel_bytes(df_dict):
    try:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            for nome_aba, df in df_dict.items():
                df.to_excel(writer, index=False, sheet_name=nome_aba[:31])
        output.seek(0)
        return output.getvalue()
    except ModuleNotFoundError:
        return None


def criar_mascara_caf_ativo(df):
    alternativas_status = [
        "st_ativo", "st_caf_ativo", "st_status_ativo", "st_registro_ativo",
        "fl_ativo", "fl_caf_ativo", "in_ativo", "ativo",
        "ds_situacao_caf", "ds_status_caf", "ds_situacao", "ds_status", "ds_situacao_unidade_familiar"
    ]
    col_status = encontrar_coluna(df, alternativas_status, obrigatoria=False)
    if col_status is None:
        raise ValueError(
            "Não foi possível identificar a coluna de situação/ativo no arquivo 01 - DADOS_CAF. "
            f"Verifique se há uma das colunas esperadas: {alternativas_status}"
        )

    valores = normalizar_texto_serie(df[col_status])
    ativos = {
        "SIM", "S", "TRUE", "1", "ATIVO", "ATIVA", "VIGENTE", "VALIDO", "VALIDA", "REGULAR"
    }
    inativos = {
        "NAO", "N", "FALSE", "0", "INATIVO", "INATIVA", "CANCELADO", "CANCELADA",
        "ENCERRADO", "ENCERRADA", "BAIXADO", "BAIXADA", "SUSPENSO", "SUSPENSA"
    }

    mascara = valores.isin(ativos)
    if mascara.sum() == 0:
        mascara = ~valores.isin(inativos) & valores.ne("") & valores.ne("NAN")
    return mascara, col_status


# =========================================================
# CARREGAMENTO
# =========================================================
@st.cache_data(show_spinner=True)
def carregar_dados_caf(origem):
    df = ler_csv_seguro(origem)
    st.write("Prévia bruta 01 - DADOS_CAF:")
    st.write(df.head())

    df = padronizar_colunas(df)

    st.write("Colunas lidas em 01 - DADOS_CAF:", df.columns.tolist())

    df = garantir_coluna(
        df,
        ["nr_caf", "nr_caf_", "caf", "numero_caf", "nu_caf", "id_caf"],
        "nr_caf"
    )

    if "nr_caf" not in df.columns:
        raise ValueError(
            f"Coluna obrigatória ausente no arquivo 01 - DADOS_CAF: nr_caf. "
            f"Colunas encontradas: {df.columns.tolist()}"
        )

    df["nr_caf"] = normalizar_nr_caf(df["nr_caf"])
    mascara_ativo, coluna_status = criar_mascara_caf_ativo(df)
    df["st_caf_ativo"] = np.where(mascara_ativo, "SIM", "NAO")

    colunas_preservar = ["nr_caf", "st_caf_ativo", coluna_status]
    if coluna_status != "st_caf_ativo":
        df = df[colunas_preservar].copy()
    else:
        df = df[["nr_caf", "st_caf_ativo"]].copy()

    return df
    

@st.cache_data(show_spinner=True)
def carregar_especificacao(origem):
    df = ler_csv_seguro(origem)
    df = padronizar_colunas(df)

    df = garantir_coluna(df, ["nr_caf", "nr_caf_"], "nr_caf")
    df = garantir_coluna(
        df,
        [
            "nm_caracterizacao_area", "ds_caracterizacao_area", "caracterizacao_area",
            "nm_especificacao", "ds_especificacao", "nm_classificacao_area"
        ],
        "nm_caracterizacao_area"
    )

    if "nr_caf" not in df.columns or "nm_caracterizacao_area" not in df.columns:
        raise ValueError(
            "Colunas obrigatórias ausentes no arquivo 03 - ESPECIFICACAO: nr_caf e nm_caracterizacao_area"
        )

    df["nr_caf"] = normalizar_nr_caf(df["nr_caf"])
    df["nm_caracterizacao_area"] = normalizar_texto_serie(df["nm_caracterizacao_area"])

    return df[["nr_caf", "nm_caracterizacao_area"]].copy()


@st.cache_data(show_spinner=True)
def carregar_area(origem):
    df = ler_csv_seguro(origem)
    df = padronizar_colunas(df)

    df = garantir_coluna(df, ["nr_area", "nr_area_"], "nr_area")
    df = garantir_coluna(df, ["nr_caf", "nr_caf_"], "nr_caf")
    df = garantir_coluna(df, ["sg_uf", "sg_uf_"], "sg_uf")
    df = garantir_coluna(df, ["nm_municipio", "nm_municipio_"], "nm_municipio")
    df = garantir_coluna(df, ["cd_municipio", "cd_municipio_"], "cd_municipio")
    df = garantir_coluna(df, ["nr_latitude", "latitude"], "nr_latitude")
    df = garantir_coluna(df, ["nr_longitude", "longitude"], "nr_longitude")
    df = garantir_coluna(df, ["st_imovel_principal", "imovel_principal"], "st_imovel_principal")
    df = garantir_coluna(df, ["ds_tipo_area", "tipo_area"], "ds_tipo_area")
    df = garantir_coluna(df, ["ds_condicao_dominio", "condicao_dominio"], "ds_condicao_dominio")
    df = garantir_coluna(df, ["ds_tipo_localizacao_area", "tipo_localizacao_area"], "ds_tipo_localizacao_area")
    df = garantir_coluna(df, ["nm_pessoa_fisica", "nome_pessoa_fisica"], "nm_pessoa_fisica")
    df = garantir_coluna(df, ["nr_cpf", "cpf"], "nr_cpf")

    obrigatorias = [
        "nr_caf", "nr_area", "sg_uf", "nm_municipio",
        "st_imovel_principal", "nr_latitude", "nr_longitude"
    ]
    faltantes = [c for c in obrigatorias if c not in df.columns]
    if faltantes:
        raise ValueError(f"Colunas obrigatórias ausentes no AREA: {faltantes}")

    df["nr_caf"] = normalizar_nr_caf(df["nr_caf"])
    df["sg_uf"] = normalizar_texto_serie(df["sg_uf"])
    df["nm_municipio"] = df["nm_municipio"].astype(str).str.strip()
    df["st_imovel_principal"] = converter_bool_sim_nao(df["st_imovel_principal"])

    df["nr_area"] = pd.to_numeric(df["nr_area"], errors="coerce")
    df["nr_latitude"] = pd.to_numeric(df["nr_latitude"], errors="coerce")
    df["nr_longitude"] = pd.to_numeric(df["nr_longitude"], errors="coerce")

    for col in ["ds_tipo_area", "ds_condicao_dominio", "ds_tipo_localizacao_area", "nm_pessoa_fisica"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    return converter_categorias(df)


@st.cache_data(show_spinner=True)
def carregar_renda(origem):
    df = ler_csv_seguro(origem)
    df = padronizar_colunas(df)

    df = garantir_coluna(df, ["nr_caf", "nr_caf_"], "nr_caf")
    df = garantir_coluna(df, ["ds_produto", "produto", "nm_produto"], "ds_produto")
    df = garantir_coluna(df, ["sg_uf", "sg_uf_"], "sg_uf")

    if "nr_caf" not in df.columns:
        raise ValueError("Coluna obrigatória ausente no arquivo PRODUCAO: nr_caf")

    df["nr_caf"] = normalizar_nr_caf(df["nr_caf"])

    if "ds_produto" in df.columns:
        produto = df["ds_produto"].astype(str).str.strip()
        produto = produto.replace("", np.nan).fillna("NÃO INFORMADO")
        df["ds_produto"] = produto
    else:
        df["ds_produto"] = "NÃO INFORMADO"

    for coluna_renda in [
        "vl_renda_auferida", "vl_renda_estimada", "vl_renda",
        "valor_renda", "renda_total", "vl_total_renda"
    ]:
        if coluna_renda in df.columns:
            df["vl_renda"] = pd.to_numeric(df[coluna_renda], errors="coerce")
            break
    else:
        df["vl_renda"] = np.nan

    if "sg_uf" in df.columns:
        df["sg_uf"] = normalizar_texto_serie(df["sg_uf"])

    return converter_categorias(df)


@st.cache_data(show_spinner=False)
def agregar_renda_por_caf(df_renda):
    if df_renda.empty:
        return pd.DataFrame(columns=["nr_caf", "renda_total_caf"])
    return (
        df_renda.groupby("nr_caf", as_index=False)["vl_renda"]
        .sum(min_count=1)
        .rename(columns={"vl_renda": "renda_total_caf"})
    )


@st.cache_data(show_spinner=False)
def agregar_produtos_por_caf(df_renda):
    if df_renda.empty:
        return pd.DataFrame(columns=["nr_caf", "ds_produto"])
    return df_renda[["nr_caf", "ds_produto"]].copy()


@st.cache_data(show_spinner=False)
def consolidar_especificacao(df_especificacao):
    if df_especificacao.empty:
        return pd.DataFrame(columns=["nr_caf", "nm_caracterizacao_area", "fl_demais_agricultores_familiares"])

    def consolidar_textos(series):
        valores = sorted({v for v in series.astype(str) if v and v != "NAN"})
        return " | ".join(valores)

    base = (
        df_especificacao.groupby("nr_caf", as_index=False)["nm_caracterizacao_area"]
        .agg(consolidar_textos)
    )
    base["fl_demais_agricultores_familiares"] = base["nm_caracterizacao_area"].str.contains(
        CARACTERIZACAO_AREA_ALVO,
        na=False
    )
    return base


@st.cache_data(show_spinner=False)
def montar_base_principal(df_caf, df_area, df_especificacao, df_renda):
    base_caf_ativo = (
        df_caf[df_caf["st_caf_ativo"] == "SIM"]
        [["nr_caf", "st_caf_ativo"]]
        .drop_duplicates(subset=["nr_caf"])
        .copy()
    )

    espec_consolidada = consolidar_especificacao(df_especificacao)
    renda_por_caf = agregar_renda_por_caf(df_renda)

    base = base_caf_ativo.merge(df_area, on="nr_caf", how="left")
    base = base.merge(espec_consolidada, on="nr_caf", how="left")
    base = base.merge(renda_por_caf, on="nr_caf", how="left")

    if "nm_caracterizacao_area" not in base.columns:
        base["nm_caracterizacao_area"] = ""
    base["nm_caracterizacao_area"] = base["nm_caracterizacao_area"].fillna("")
    base["fl_demais_agricultores_familiares"] = base.get("fl_demais_agricultores_familiares", False).fillna(False)
    base["renda_total_caf"] = pd.to_numeric(base.get("renda_total_caf", 0), errors="coerce").fillna(0)

    return converter_categorias(base)


@st.cache_data(show_spinner=False)
def montar_base_produtos(df_base_ativa, df_renda):
    if df_renda.empty:
        return pd.DataFrame(columns=["nr_caf", "sg_uf", "nm_municipio", "ds_produto"])

    base = (
        df_base_ativa[df_base_ativa["st_imovel_principal"].astype(str) == "SIM"]
        [["nr_caf", "sg_uf", "nm_municipio"]]
        .drop_duplicates(subset=["nr_caf", "sg_uf", "nm_municipio"])
        .copy()
    )
    base = base.merge(agregar_produtos_por_caf(df_renda), on="nr_caf", how="inner")
    base["ds_produto"] = base["ds_produto"].astype(str).replace("", "NÃO INFORMADO").fillna("NÃO INFORMADO")
    return base


@st.cache_data(show_spinner=False)
def diagnostico_area(df):
    ufs_validas = {
        "AC","AL","AP","AM","BA","CE","DF","ES","GO","MA","MT","MS",
        "MG","PA","PB","PR","PE","PI","RJ","RN","RS","RO","RR","SC",
        "SP","SE","TO"
    }

    coords_validas = (
        df["nr_latitude"].between(-35, 10, inclusive="both") &
        df["nr_longitude"].between(-75, -30, inclusive="both")
    )

    nulos = df.isna().sum().sort_values(ascending=False)
    nulos = nulos[nulos > 0]

    return {
        "linhas": len(df),
        "colunas": len(df.columns),
        "ufs_invalidas": df.loc[~df["sg_uf"].isin(ufs_validas), "sg_uf"].dropna().astype(str).unique().tolist(),
        "qtd_ufs_invalidas_linhas": int((~df["sg_uf"].isin(ufs_validas)).sum()),
        "area_nula_ou_negativa": int(((df["nr_area"].isna()) | (df["nr_area"] <= 0)).sum()),
        "caf_duplicado": int(df["nr_caf"].duplicated(keep=False).sum()),
        "coords_validas": int(coords_validas.fillna(False).sum()),
        "coords_invalidas": int((~coords_validas.fillna(False)).sum()),
        "nulos": nulos
    }


# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.title("Painel de Controle")

usar_upload = st.sidebar.toggle("Usar upload de arquivos", value=False)

if usar_upload:
    arquivo_dados_caf = st.sidebar.file_uploader("Arquivo 01 - DADOS_CAF", type=["csv"])
    arquivo_especificacao = st.sidebar.file_uploader("Arquivo 03 - ESPECIFICACAO", type=["csv"])
    arquivo_area = st.sidebar.file_uploader("Arquivo 05 - AREA", type=["csv"])
    arquivo_renda = st.sidebar.file_uploader("Arquivo 07 - PRODUCAO", type=["csv"])
else:
    arquivo_dados_caf = ARQUIVO_DADOS_CAF_PADRAO
    arquivo_especificacao = ARQUIVO_ESPECIFICACAO_PADRAO
    arquivo_area = ARQUIVO_AREA_PADRAO
    arquivo_renda = ARQUIVO_RENDA_PADRAO

if not arquivo_dados_caf:
    st.info("Envie o arquivo 01 - DADOS_CAF na barra lateral para iniciar.")
    st.stop()

if not arquivo_especificacao:
    st.info("Envie o arquivo 03 - ESPECIFICACAO na barra lateral para iniciar.")
    st.stop()

if not arquivo_area:
    st.info("Envie o arquivo 05 - AREA na barra lateral para iniciar.")
    st.stop()

# =========================================================
# LEITURA
# =========================================================
try:
    df_dados_caf = carregar_dados_caf(arquivo_dados_caf)
except Exception as e:
    st.error(f"Erro ao carregar o arquivo 01 - DADOS_CAF: {e}")
    st.stop()

try:
    df_especificacao = carregar_especificacao(arquivo_especificacao)
except Exception as e:
    st.error(f"Erro ao carregar o arquivo 03 - ESPECIFICACAO: {e}")
    st.stop()

try:
    df_area = carregar_area(arquivo_area)
except Exception as e:
    st.error(f"Erro ao carregar o arquivo 05 - AREA: {e}")
    st.stop()

try:
    if arquivo_renda:
        df_renda = carregar_renda(arquivo_renda)
        renda_disponivel = True
    else:
        df_renda = pd.DataFrame(columns=["nr_caf", "ds_produto", "vl_renda"])
        renda_disponivel = False
except Exception as e:
    st.warning(f"Arquivo 07 - PRODUCAO não carregado ou com problema: {e}")
    df_renda = pd.DataFrame(columns=["nr_caf", "ds_produto", "vl_renda"])
    renda_disponivel = False

# Mantém apenas CAFs ativos em todas as bases auxiliares
cafs_ativos = set(df_dados_caf.loc[df_dados_caf["st_caf_ativo"] == "SIM", "nr_caf"].astype(str))

df_area = df_area[df_area["nr_caf"].astype(str).isin(cafs_ativos)].copy()
df_especificacao = df_especificacao[df_especificacao["nr_caf"].astype(str).isin(cafs_ativos)].copy()
df_renda = df_renda[df_renda["nr_caf"].astype(str).isin(cafs_ativos)].copy()

df_base = montar_base_principal(df_dados_caf, df_area, df_especificacao, df_renda)

# =========================================================
# FILTROS
# =========================================================
st.sidebar.markdown("### Filtros")
st.sidebar.caption("Todos os filtros consideram somente CAFs ativos.")

ufs = sorted([x for x in df_base["sg_uf"].dropna().astype(str).unique().tolist() if x and x != "NAN"])
uf_sel = st.sidebar.multiselect("Estado (UF)", options=ufs, placeholder="Selecione as opções")

auditoria_filtros = []

def registrar_etapa_filtro(nome, df_ref):
    auditoria_filtros.append({
        "etapa": nome,
        "linhas": len(df_ref),
        "cafs_unicos": contar_cafs_unicos(df_ref)
    })


df_filtrado = df_base.copy()
registrar_etapa_filtro("Base ativa inicial", df_filtrado)

if uf_sel:
    df_filtrado = df_filtrado[df_filtrado["sg_uf"].astype(str).isin(uf_sel)]
registrar_etapa_filtro("Após filtro UF", df_filtrado)

municipios = sorted([x for x in df_filtrado["nm_municipio"].dropna().astype(str).unique().tolist() if x and x.lower() != "nan"])
municipio_sel = st.sidebar.multiselect("Município", options=municipios, placeholder="Selecione as opções")
if municipio_sel:
    df_filtrado = df_filtrado[df_filtrado["nm_municipio"].astype(str).isin(municipio_sel)]
registrar_etapa_filtro("Após filtro município", df_filtrado)

if "ds_tipo_area" in df_filtrado.columns:
    tipos_area = sorted([x for x in df_filtrado["ds_tipo_area"].dropna().astype(str).unique().tolist() if x and x.lower() != "nan"])
    tipo_area_sel = st.sidebar.multiselect("Tipo de área", options=tipos_area, placeholder="Selecione as opções")
    if tipo_area_sel:
        df_filtrado = df_filtrado[df_filtrado["ds_tipo_area"].astype(str).isin(tipo_area_sel)]
    registrar_etapa_filtro("Após filtro tipo de área", df_filtrado)
else:
    tipo_area_sel = []

if "ds_condicao_dominio" in df_filtrado.columns:
    dominio = sorted([x for x in df_filtrado["ds_condicao_dominio"].dropna().astype(str).unique().tolist() if x and x.lower() != "nan"])
    dominio_sel = st.sidebar.multiselect("Condição de domínio", options=dominio, placeholder="Selecione as opções")
    if dominio_sel:
        df_filtrado = df_filtrado[df_filtrado["ds_condicao_dominio"].astype(str).isin(dominio_sel)]
    registrar_etapa_filtro("Após filtro condição de domínio", df_filtrado)
else:
    dominio_sel = []

if "nm_caracterizacao_area" in df_filtrado.columns:
    caracterizacoes = sorted([x for x in df_filtrado["nm_caracterizacao_area"].dropna().astype(str).unique().tolist() if x and x.lower() != "nan"])
    caracterizacao_sel = st.sidebar.multiselect("Caracterização da área", options=caracterizacoes, placeholder="Selecione as opções")
    if caracterizacao_sel:
        mascara_caracterizacao = pd.Series(False, index=df_filtrado.index)
        for valor in caracterizacao_sel:
            mascara_caracterizacao = mascara_caracterizacao | df_filtrado["nm_caracterizacao_area"].astype(str).str.contains(valor, na=False)
        df_filtrado = df_filtrado[mascara_caracterizacao]
    registrar_etapa_filtro("Após filtro caracterização", df_filtrado)
else:
    caracterizacao_sel = []

imovel_principal_sel = st.sidebar.selectbox("Imóvel principal", ["Todos", "SIM", "NAO"])
if imovel_principal_sel != "Todos":
    df_filtrado = df_filtrado[df_filtrado["st_imovel_principal"].astype(str) == imovel_principal_sel]
registrar_etapa_filtro("Após filtro imóvel principal", df_filtrado)

somente_com_coordenadas = st.sidebar.checkbox("Apenas coordenadas válidas", value=False)
if somente_com_coordenadas:
    coords_validas_filtro = (
        df_filtrado["nr_latitude"].between(-35, 10, inclusive="both") &
        df_filtrado["nr_longitude"].between(-75, -30, inclusive="both")
    )
    df_filtrado = df_filtrado[coords_validas_filtro]
registrar_etapa_filtro("Após filtro coordenadas válidas", df_filtrado)

df_auditoria_filtros = pd.DataFrame(auditoria_filtros)

# drill-down
st.sidebar.markdown("### Drill-down")
drill_uf = st.sidebar.selectbox("UF para detalhamento", ["Todas"] + ufs)
if drill_uf != "Todas":
    df_drill = df_filtrado[df_filtrado["sg_uf"].astype(str) == drill_uf].copy()
else:
    df_drill = df_filtrado.copy()

municipios_drill = sorted(df_drill["nm_municipio"].dropna().astype(str).unique().tolist())
drill_municipio = st.sidebar.selectbox("Município para detalhamento", ["Todos"] + municipios_drill)
if drill_municipio != "Todos":
    df_drill = df_drill[df_drill["nm_municipio"].astype(str) == drill_municipio].copy()

if len(df_filtrado) > MAX_REGISTROS_SEM_FILTRO and not (uf_sel or municipio_sel):
    st.warning(
        f"A base possui {formatar_inteiro(len(df_filtrado))} registros após os filtros atuais. "
        "Selecione ao menos uma UF ou município para continuar com segurança."
    )
    st.stop()

# =========================================================
# CABEÇALHO
# =========================================================
st.markdown("# 📊 PROJETO CADERNOS DO CAF")
st.markdown("### BI Premium - Relatório Analítico")
st.markdown(
    f"<div class='small-note'>Registros após filtros: {formatar_inteiro(len(df_filtrado))} | "
    f"CAFs únicos após filtros: {formatar_inteiro(contar_cafs_unicos(df_filtrado))} | "
    f"Base de drill-down: {formatar_inteiro(len(df_drill))} | "
    f"Regra global ativa: somente CAFs ativos</div>",
    unsafe_allow_html=True
)

if somente_com_coordenadas:
    st.warning("O filtro 'Apenas coordenadas válidas' está ligado. Isso reduz a contagem de CAFs para somente os registros com latitude e longitude válidas.")

# =========================================================
# KPIs
# =========================================================
base_unica_caf = df_filtrado.drop_duplicates(subset=["nr_caf"]).copy()
base_area_kpi = base_unica_caf[base_unica_caf["fl_demais_agricultores_familiares"] == True].copy()

total_cafs = base_unica_caf["nr_caf"].nunique()
total_area = base_area_kpi["nr_area"].sum(min_count=1)
total_renda = base_unica_caf["renda_total_caf"].sum(min_count=1)
total_ufs = df_filtrado["sg_uf"].nunique()
total_municipios = df_filtrado["cd_municipio"].nunique() if "cd_municipio" in df_filtrado.columns else df_filtrado["nm_municipio"].nunique()
qtd_imovel_principal = (df_filtrado["st_imovel_principal"].astype(str) == "SIM").sum()
area_media = base_area_kpi["nr_area"].mean()
coords_validas = (
    df_filtrado["nr_latitude"].between(-35, 10, inclusive="both") &
    df_filtrado["nr_longitude"].between(-75, -30, inclusive="both")
).fillna(False).sum()

k1, k2, k3, k4 = st.columns(4)
with k1:
    criar_card_kpi("Total de CAFs", formatar_inteiro(total_cafs), "CAFs únicos ativos")
with k2:
    criar_card_kpi("Área Total (ha)", formatar_decimal_br(total_area), "CAFs ativos com 'Demais Agricultores Familiares'")
with k3:
    criar_card_kpi("Renda Total (R$)", formatar_decimal_br(total_renda), "Somente CAFs ativos, sem duplicidade")
with k4:
    criar_card_kpi("Municípios", formatar_inteiro(total_municipios), "Municípios distintos")

k5, k6, k7, k8 = st.columns(4)
with k5:
    criar_card_kpi("UFs", formatar_inteiro(total_ufs), "Unidades da federação")
with k6:
    criar_card_kpi("Imóveis principais", formatar_inteiro(qtd_imovel_principal), "Marcados como SIM")
with k7:
    criar_card_kpi("Área média (ha)", formatar_decimal_br(area_media), "Sobre base da área elegível")
with k8:
    criar_card_kpi("Coordenadas válidas", formatar_inteiro(coords_validas), "Registros mapeáveis")

# =========================================================
# ABAS
# =========================================================
tab_visao, tab_comparativo, tab_mapa, tab_graficos, tab_produtos, tab_tabela, tab_diagnostico = st.tabs(
    ["📌 Visão Geral", "📊 Comparativo UFs", "🗺️ Mapa", "📈 Gráficos", "🥇 Produtos", "📋 Tabela", "🧪 Diagnóstico"]
)

# =========================================================
# VISÃO GERAL
# =========================================================
with tab_visao:
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Top 10 UFs por área total")
        top_uf_area = (
            df_filtrado[df_filtrado["fl_demais_agricultores_familiares"] == True]
            .groupby("sg_uf", as_index=False, observed=True)["nr_area"]
            .sum()
            .sort_values("nr_area", ascending=False)
            .head(10)
        )

        if not top_uf_area.empty:
            fig = px.bar(
                top_uf_area,
                x="sg_uf",
                y="nr_area",
                text="nr_area",
                color="nr_area",
                color_continuous_scale="Blues"
            )
            fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
            fig.update_layout(
                **tema_plotly_claro(),
                coloraxis_showscale=False,
                xaxis_title="UF",
                yaxis_title="Área Total (ha)"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Sem dados elegíveis para área total com a caracterização solicitada.")

    with col_b:
        st.markdown("#### Top 10 municípios por quantidade de CAF")
        top_mun_caf = (
            df_filtrado.groupby("nm_municipio", as_index=False, observed=True)["nr_caf"]
            .nunique()
            .rename(columns={"nr_caf": "qtd_caf"})
            .sort_values("qtd_caf", ascending=False)
            .head(10)
        )

        if not top_mun_caf.empty:
            fig = px.bar(
                top_mun_caf,
                x="qtd_caf",
                y="nm_municipio",
                orientation="h",
                text="qtd_caf",
                color="qtd_caf",
                color_continuous_scale="Teal"
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(
                **tema_plotly_claro(),
                coloraxis_showscale=False,
                xaxis_title="Quantidade de CAFs",
                yaxis_title="Município",
                yaxis={"categoryorder": "total ascending"}
            )
            st.plotly_chart(fig, use_container_width=True)

# =========================================================
# COMPARATIVO UFS
# =========================================================
with tab_comparativo:
    st.markdown("#### Comparativo executivo por UF")

    comparativo_uf = (
        base_unica_caf.groupby("sg_uf", as_index=False, observed=True)
        .agg(
            qtd_caf=("nr_caf", "nunique"),
            area_total=("nr_area", lambda s: s[base_unica_caf.loc[s.index, "fl_demais_agricultores_familiares"] == True].sum()),
            renda_total=("renda_total_caf", "sum"),
            qtd_municipios=("nm_municipio", "nunique")
        )
        .sort_values("qtd_caf", ascending=False)
    )

    if not comparativo_uf.empty:
        metrica_comp = st.selectbox(
            "Métrica do comparativo",
            ["qtd_caf", "area_total", "renda_total", "qtd_municipios"],
            format_func=lambda x: {
                "qtd_caf": "Quantidade de CAFs",
                "area_total": "Área Total",
                "renda_total": "Renda Total",
                "qtd_municipios": "Quantidade de Municípios"
            }[x]
        )

        fig = px.bar(
            comparativo_uf.sort_values(metrica_comp, ascending=False).head(15),
            x="sg_uf",
            y=metrica_comp,
            text=metrica_comp,
            color=metrica_comp,
            color_continuous_scale="Cividis"
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(
            **tema_plotly_claro(),
            coloraxis_showscale=False,
            xaxis_title="UF",
            yaxis_title=""
        )
        st.plotly_chart(fig, use_container_width=True)

        comp_exibir = comparativo_uf.copy()
        if comp_exibir["qtd_caf"].sum() > 0:
            comp_exibir["participacao_caf_%"] = (comp_exibir["qtd_caf"] / comp_exibir["qtd_caf"].sum() * 100).round(2)
        else:
            comp_exibir["participacao_caf_%"] = 0

        if comp_exibir["area_total"].sum() > 0:
            comp_exibir["participacao_area_%"] = (comp_exibir["area_total"] / comp_exibir["area_total"].sum() * 100).round(2)
        else:
            comp_exibir["participacao_area_%"] = 0

        st.markdown("##### Tabela comparativa")
        st.dataframe(comp_exibir, use_container_width=True, hide_index=True)

# =========================================================
# MAPA
# =========================================================
with tab_mapa:
    st.markdown("#### Mapa dos imóveis - tema claro")

    mapa_df = df_filtrado.copy()
    mapa_df = mapa_df.dropna(subset=["nr_latitude", "nr_longitude"])
    mapa_df = mapa_df[
        mapa_df["nr_latitude"].between(-35, 10, inclusive="both") &
        mapa_df["nr_longitude"].between(-75, -30, inclusive="both")
    ]

    if mapa_df.empty:
        st.info("Não há coordenadas válidas para exibir no mapa.")
    else:
        if len(mapa_df) > MAX_PONTOS_MAPA:
            mapa_df = mapa_df.sample(MAX_PONTOS_MAPA, random_state=42)
            st.warning(f"O mapa foi limitado a {formatar_inteiro(MAX_PONTOS_MAPA)} pontos para melhorar a performance.")

        mapa_df["faixa_area"] = pd.cut(
            mapa_df["nr_area"],
            bins=[-np.inf, 5, 20, 50, 100, np.inf],
            labels=["Até 5 ha", "5 a 20 ha", "20 a 50 ha", "50 a 100 ha", "Acima de 100 ha"]
        )

        fig_map = px.scatter_mapbox(
            mapa_df,
            lat="nr_latitude",
            lon="nr_longitude",
            color="faixa_area" if "faixa_area" in mapa_df.columns else None,
            size="nr_area",
            size_max=18,
            hover_name="nm_municipio" if "nm_municipio" in mapa_df.columns else None,
            hover_data={
                "nr_caf": True,
                "sg_uf": True,
                "nr_area": True,
                "st_imovel_principal": True,
                "nm_caracterizacao_area": True,
                "nr_latitude": False,
                "nr_longitude": False
            },
            zoom=3,
            height=720
        )
        fig_map.update_layout(
            mapbox_style="carto-positron",
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
            legend_title_text="Faixa de área"
        )
        st.plotly_chart(fig_map, use_container_width=True)

# =========================================================
# GRÁFICOS
# =========================================================
with tab_graficos:
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### Distribuição por tipo de área")
        if "ds_tipo_area" in df_filtrado.columns:
            dist_tipo_area = (
                df_filtrado.groupby("ds_tipo_area", as_index=False, observed=True)["nr_caf"]
                .nunique()
                .rename(columns={"nr_caf": "qtd_caf"})
                .sort_values("qtd_caf", ascending=False)
                .head(15)
            )
            if not dist_tipo_area.empty:
                fig = px.bar(
                    dist_tipo_area,
                    x="qtd_caf",
                    y="ds_tipo_area",
                    orientation="h",
                    text="qtd_caf",
                    color="qtd_caf",
                    color_continuous_scale="Purples"
                )
                fig.update_layout(
                    **tema_plotly_claro(),
                    coloraxis_showscale=False,
                    xaxis_title="Quantidade de CAFs",
                    yaxis_title="Tipo de área",
                    yaxis={"categoryorder": "total ascending"}
                )
                st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("#### Imóvel principal: SIM x NÃO")
        dist_principal = (
            df_filtrado.groupby("st_imovel_principal", as_index=False, observed=True)["nr_caf"]
            .nunique()
            .rename(columns={"nr_caf": "qtd_caf"})
        )
        if not dist_principal.empty:
            fig = px.pie(
                dist_principal,
                names="st_imovel_principal",
                values="qtd_caf",
                hole=0.55
            )
            fig.update_layout(**tema_plotly_claro(), legend_title_text="Imóvel principal")
            st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)

    with c3:
        st.markdown("#### Drill-down por município")
        drill_mun = (
            df_drill.groupby("nm_municipio", as_index=False, observed=True)["nr_caf"]
            .nunique()
            .rename(columns={"nr_caf": "qtd_caf"})
            .sort_values("qtd_caf", ascending=False)
            .head(15)
        )
        if not drill_mun.empty:
            fig = px.bar(
                drill_mun,
                x="qtd_caf",
                y="nm_municipio",
                orientation="h",
                text="qtd_caf",
                color="qtd_caf",
                color_continuous_scale="Mint"
            )
            fig.update_layout(
                **tema_plotly_claro(),
                coloraxis_showscale=False,
                xaxis_title="Quantidade de CAFs",
                yaxis_title="Município",
                yaxis={"categoryorder": "total ascending"}
            )
            st.plotly_chart(fig, use_container_width=True)

    with c4:
        st.markdown("#### Top 10 municípios por área total")
        top_mun_area = (
            df_filtrado[df_filtrado["fl_demais_agricultores_familiares"] == True]
            .groupby("nm_municipio", as_index=False, observed=True)["nr_area"]
            .sum()
            .sort_values("nr_area", ascending=False)
            .head(10)
        )
        if not top_mun_area.empty:
            fig = px.bar(
                top_mun_area,
                x="nr_area",
                y="nm_municipio",
                orientation="h",
                text="nr_area",
                color="nr_area",
                color_continuous_scale="Sunset"
            )
            fig.update_layout(
                **tema_plotly_claro(),
                coloraxis_showscale=False,
                xaxis_title="Área Total (ha)",
                yaxis_title="Município",
                yaxis={"categoryorder": "total ascending"}
            )
            st.plotly_chart(fig, use_container_width=True)

# =========================================================
# PRODUTOS
# =========================================================
with tab_produtos:
    st.markdown("#### Ranking de produtos declarados")
    if not renda_disponivel:
        st.warning("Esta aba depende do arquivo 07 - PRODUCAO.")
    else:
        base_produtos = montar_base_produtos(df_base, df_renda)

        if uf_sel:
            base_produtos = base_produtos[base_produtos["sg_uf"].astype(str).isin(uf_sel)]
        if municipio_sel:
            base_produtos = base_produtos[base_produtos["nm_municipio"].astype(str).isin(municipio_sel)]

        if base_produtos.empty:
            st.info("Não há dados suficientes para exibir o ranking de produtos.")
        else:
            c1, c2 = st.columns([1, 1])
            with c1:
                ufs_produtos = sorted(base_produtos["sg_uf"].dropna().astype(str).unique().tolist())
                uf_rank_sel = st.selectbox("Selecione a UF", options=ufs_produtos)
            with c2:
                top_n = st.slider("Quantidade de produtos no ranking", min_value=5, max_value=20, value=10)

            top_produtos = (
                base_produtos[base_produtos["sg_uf"].astype(str) == uf_rank_sel]
                .groupby("ds_produto", as_index=False)
                .size()
                .rename(columns={"size": "qtd_declaracoes"})
                .sort_values("qtd_declaracoes", ascending=False)
                .head(top_n)
            )

            total_decl = top_produtos["qtd_declaracoes"].sum()
            top_produtos["percentual_%"] = np.where(
                total_decl > 0,
                (top_produtos["qtd_declaracoes"] / total_decl * 100).round(2),
                0
            )

            p1, p2 = st.columns([1.6, 1])

            with p1:
                fig = px.bar(
                    top_produtos,
                    x="qtd_declaracoes",
                    y="ds_produto",
                    orientation="h",
                    text="percentual_%",
                    color="qtd_declaracoes",
                    color_continuous_scale="Viridis"
                )
                fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
                fig.update_layout(
                    **tema_plotly_claro(),
                    coloraxis_showscale=False,
                    xaxis_title="Quantidade de declarações",
                    yaxis_title="Produto",
                    yaxis={"categoryorder": "total ascending"}
                )
                st.plotly_chart(fig, use_container_width=True)

            with p2:
                st.markdown("##### Tabela do ranking")
                st.dataframe(top_produtos, use_container_width=True, hide_index=True)

# =========================================================
# TABELA
# =========================================================
with tab_tabela:
    st.markdown("#### Base filtrada")

    colunas_exibir = [
        c for c in [
            "nr_caf", "sg_uf", "nm_municipio", "nr_area",
            "ds_tipo_area", "ds_condicao_dominio",
            "ds_tipo_localizacao_area", "st_imovel_principal",
            "nm_caracterizacao_area", "fl_demais_agricultores_familiares",
            "nr_latitude", "nr_longitude", "renda_total_caf"
        ] if c in df_filtrado.columns
    ]

    tabela = df_filtrado[colunas_exibir].copy()

    if "nr_area" in tabela.columns:
        tabela["nr_area"] = tabela["nr_area"].round(2)
    if "renda_total_caf" in tabela.columns:
        tabela["renda_total_caf"] = tabela["renda_total_caf"].round(2)

    total_linhas = len(tabela)
    total_paginas = max(1, math.ceil(total_linhas / MAX_LINHAS_TABELA_POR_PAGINA))

    c_pag1, c_pag2, c_pag3 = st.columns([1, 1, 3])
    with c_pag1:
        pagina = st.number_input("Página", min_value=1, max_value=total_paginas, value=1, step=1)
    with c_pag2:
        st.metric("Total de páginas", formatar_inteiro(total_paginas))
    with c_pag3:
        st.markdown(
            f"<div class='small-note'>Exibindo {formatar_inteiro(MAX_LINHAS_TABELA_POR_PAGINA)} linhas por página.</div>",
            unsafe_allow_html=True
        )

    inicio = (pagina - 1) * MAX_LINHAS_TABELA_POR_PAGINA
    fim = inicio + MAX_LINHAS_TABELA_POR_PAGINA
    tabela_pagina = tabela.iloc[inicio:fim]

    st.dataframe(tabela_pagina, use_container_width=True, height=520, hide_index=True)

    if total_linhas > MAX_EXPORT_LINHAS:
        st.warning(f"A exportação será limitada a {formatar_inteiro(MAX_EXPORT_LINHAS)} linhas.")
        tabela_export = tabela.head(MAX_EXPORT_LINHAS)
    else:
        tabela_export = tabela

    csv_export = tabela_export.to_csv(index=False, sep=";", encoding="utf-8-sig").encode("utf-8-sig")
    excel_export = para_excel_bytes({"base_filtrada": tabela_export})

    cexp1, cexp2 = st.columns(2)
    with cexp1:
        st.download_button(
            label="Baixar CSV",
            data=csv_export,
            file_name="base_filtrada_dashboard.csv",
            mime="text/csv"
        )
    with cexp2:
        if excel_export is not None:
            st.download_button(
                label="Baixar Excel",
                data=excel_export,
                file_name="base_filtrada_dashboard.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        else:
            st.info("Exportação em Excel indisponível. Instale openpyxl para habilitar.")

# =========================================================
# DIAGNÓSTICO
# =========================================================
with tab_diagnostico:
    st.markdown("#### Diagnóstico da base AREA")
    diag = diagnostico_area(df_area)

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Linhas", formatar_inteiro(diag["linhas"]))
    d2.metric("Colunas", formatar_inteiro(diag["colunas"]))
    d3.metric("Área nula/zero/negativa", formatar_inteiro(diag["area_nula_ou_negativa"]))
    d4.metric("CAF duplicado", formatar_inteiro(diag["caf_duplicado"]))

    d5, d6, d7 = st.columns(3)
    d5.metric("Linhas com UF inválida", formatar_inteiro(diag["qtd_ufs_invalidas_linhas"]))
    d6.metric("Coordenadas válidas", formatar_inteiro(diag["coords_validas"]))
    d7.metric("CAFs ativos", formatar_inteiro(len(cafs_ativos)))

    if diag["ufs_invalidas"]:
        st.warning(f"UFs inválidas encontradas: {diag['ufs_invalidas']}")
    else:
        st.success("Nenhuma UF inválida encontrada.")

    if not diag["nulos"].empty:
        st.markdown("##### Colunas com valores nulos")
        st.dataframe(
            diag["nulos"].reset_index().rename(columns={"index": "coluna", 0: "qtd_nulos"}),
            use_container_width=True,
            hide_index=True
        )

    st.markdown("##### Auditoria das regras")
    coluna_status_ativo = [c for c in df_dados_caf.columns if c not in ["nr_caf", "st_caf_ativo"]]
    coluna_status_ativo = coluna_status_ativo[0] if coluna_status_ativo else "st_caf_ativo"

    coords_area_validas = (
        df_area["nr_latitude"].between(-35, 10, inclusive="both") &
        df_area["nr_longitude"].between(-75, -30, inclusive="both")
    ) if not df_area.empty else pd.Series(dtype=bool)

    auditoria_origem = pd.DataFrame([
        {"etapa": "01 - DADOS_CAF (total)", "linhas": len(df_dados_caf), "cafs_unicos": contar_cafs_unicos(df_dados_caf)},
        {"etapa": "01 - DADOS_CAF (ativos)", "linhas": len(df_dados_caf[df_dados_caf["st_caf_ativo"] == "SIM"]), "cafs_unicos": len(cafs_ativos)},
        {"etapa": "05 - AREA com CAF ativo", "linhas": len(df_area), "cafs_unicos": contar_cafs_unicos(df_area)},
        {"etapa": "05 - AREA sem correspondência", "linhas": 0, "cafs_unicos": max(len(cafs_ativos) - contar_cafs_unicos(df_area), 0)},
        {"etapa": "05 - AREA com coordenadas válidas", "linhas": int(coords_area_validas.fillna(False).sum()) if len(coords_area_validas) else 0, "cafs_unicos": df_area.loc[coords_area_validas.fillna(False), "nr_caf"].astype(str).nunique() if len(coords_area_validas) else 0},
        {"etapa": "07 - PRODUCAO com CAF ativo", "linhas": len(df_renda), "cafs_unicos": contar_cafs_unicos(df_renda)},
        {"etapa": "Base principal montada", "linhas": len(df_base), "cafs_unicos": contar_cafs_unicos(df_base)},
        {"etapa": "Base após filtros atuais", "linhas": len(df_filtrado), "cafs_unicos": contar_cafs_unicos(df_filtrado)},
    ])

    st.write("Coluna usada para identificar CAF ativo no 01 - DADOS_CAF:", coluna_status_ativo)
    st.dataframe(auditoria_origem, use_container_width=True, hide_index=True)

    st.markdown("##### Impacto de cada filtro")
    st.dataframe(df_auditoria_filtros, use_container_width=True, hide_index=True)

    st.write("Total de CAFs (KPI):", total_cafs)
    st.write("Total de CAFs recalculado:", base_unica_caf["nr_caf"].nunique())
    st.write("Área total (KPI):", total_area)
    st.write("Área total recalculada pela regra da caracterização:", base_area_kpi["nr_area"].sum())
    st.write("Renda total (KPI):", total_renda)
    st.write("Renda total sem duplicar CAF ativo:", base_unica_caf["renda_total_caf"].sum())

    with st.expander("Verificação rápida do motivo da diferença de contagem"):
        st.write("1. Compare o número de '01 - DADOS_CAF (ativos)' com 'Base após filtros atuais'.")
        st.write("2. Se o número cair apenas quando o filtro de coordenadas estiver ligado, a diferença vem das coordenadas inválidas ou ausentes.")
        st.write("3. Se cair já em '05 - AREA com CAF ativo', faltam CAFs ativos sem registro correspondente no arquivo de área.")
        st.write("4. O KPI 'Total de CAFs' usa CAF único da base filtrada. O cabeçalho mostra também 'CAFs únicos após filtros'.")

    if renda_disponivel:
        st.markdown("##### Estrutura da base de PRODUCAO")
        st.write(f"Linhas: **{formatar_inteiro(len(df_renda))}**")
        st.write(f"Colunas: **{formatar_inteiro(len(df_renda.columns))}**")
        st.code("\n".join(df_renda.columns), language="text")

# =========================================================
# RODAPÉ
# =========================================================
st.divider()
st.caption(
    "Dashboard BI Premium desenvolvido em Streamlit para análise dos arquivos do Projeto Cadernos do CAF, "
    "com regra global de CAF ativo aplicada aos filtros, KPIs e visualizações."
)
