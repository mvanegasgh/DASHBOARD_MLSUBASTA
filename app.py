"""
Dashboard de Analítica Predictiva - Subasta Ganadera Suganorte S.A. (Zarzal, Valle)
Interfaz Institucional Oficial alineada a la marca Suganorte S.A.
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt

# =========================================================
# CONFIGURACIÓN DE PÁGINA
# =========================================================
st.set_page_config(
    page_title="Suganorte S.A. | Analítica Predictiva",
    page_icon="🐂",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# RECURSOS DE MARCA (LOGOS EN SVG VECTORIAL SEGURO)
# =========================================================
LOGO_COLOR_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 450 110" width="100%">
  <path d="M 10,70 Q 25,20 40,70 T 70,70" fill="none" stroke="#008037" stroke-width="8" stroke-linecap="round"/>
  <text x="15" y="85" font-family="'Inter', sans-serif" font-weight="900" font-size="52" fill="#0B2265" letter-spacing="-2">Suganorte</text>
  <text x="325" y="85" font-family="'Inter', sans-serif" font-weight="800" font-size="42" fill="#0B2265">S.A.</text>
  <rect x="15" y="94" width="70" height="6" fill="#FFD100"/>
  <rect x="85" y="94" width="70" height="6" fill="#008037"/>
  <rect x="155" y="94" width="40" height="6" fill="#E11D48"/>
  <text x="15" y="108" font-family="'Inter', sans-serif" font-weight="600" font-size="11" fill="#008037">Líderes en comercialización ganadera en el Suroccidente</text>
  <g transform="translate(390, 25) rotate(25)">
    <rect x="0" y="0" width="12" height="35" rx="3" fill="#1E293B"/>
    <rect x="-8" y="-10" width="28" height="14" rx="2" fill="#1E293B"/>
  </g>
</svg>
"""

LOGO_WHITE_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 450 110" height="60">
  <path d="M 10,70 Q 25,20 40,70 T 70,70" fill="none" stroke="#FFFFFF" stroke-width="8" stroke-linecap="round"/>
  <text x="15" y="85" font-family="'Inter', sans-serif" font-weight="900" font-size="52" fill="#FFFFFF" letter-spacing="-2">Suganorte</text>
  <text x="325" y="85" font-family="'Inter', sans-serif" font-weight="800" font-size="42" fill="#FFFFFF">S.A.</text>
  <g transform="translate(390, 25) rotate(25)">
    <rect x="0" y="0" width="12" height="35" rx="3" fill="#FFFFFF"/>
    <rect x="-8" y="-10" width="28" height="14" rx="2" fill="#FFFFFF"/>
  </g>
</svg>
"""

# =========================================================
# ESTILOS CSS AGRESIVOS (SOBREESCRIBEN EL TEMA DE STREAMLIT)
# =========================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    /* LIMPIEZA TOTAL DE FONDOS Y TEMA ANTERIOR */
    html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"], [data-testid="stMain"] {
        background-color: #F8FAFC !important;
        color: #0F172A !important;
        font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
    }

    h1, h2, h3, h4, h5, h6, p, span, label, input, button {
        font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
    }

    /* BANNER INSTITUCIONAL */
    .suganorte-header-container {
        background: #0B2265;
        border-radius: 12px;
        overflow: hidden;
        margin-bottom: 24px;
        box-shadow: 0 4px 15px rgba(11, 34, 101, 0.15);
    }
    .tricolor-stripe {
        height: 6px;
        background: linear-gradient(90deg, #FFD100 0% 33%, #008037 33% 66%, #E11D48 66% 100%);
        width: 100%;
    }
    .suganorte-banner-body {
        padding: 20px 28px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 20px;
    }
    .suganorte-title {
        color: #FFFFFF !important;
        margin: 0 !important;
        font-size: 1.65rem !important;
        font-weight: 800 !important;
        letter-spacing: -0.02em;
    }
    .suganorte-subtitle {
        color: #FFD100 !important;
        margin-top: 4px !important;
        margin-bottom: 0 !important;
        font-size: 0.9rem !important;
        font-weight: 600 !important;
    }

    /* BARRA LATERAL */
    section[data-testid="stSidebar"], [data-testid="stSidebarContent"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E2E8F0 !important;
    }

    /* REMOVER TONOS BEIGE/CREMA DE COMPONENTES NATIVOS */
    div[data-baseweb="select"] > div, 
    div[data-baseweb="input"] > div,
    input, 
    div[role="listbox"] {
        background-color: #FFFFFF !important;
        border-color: #CBD5E1 !important;
        color: #0F172A !important;
        border-radius: 8px !important;
    }

    /* ETIQUETAS MULTISELECT EN AZUL REY */
    span[data-baseweb="tag"] {
        background-color: #0B2265 !important;
        border-radius: 6px !important;
        border: none !important;
    }
    span[data-baseweb="tag"] span {
        color: #FFFFFF !important;
        font-weight: 600 !important;
    }

    /* METRICAS */
    div[data-testid="stMetric"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-top: 4px solid #008037 !important;
        border-radius: 10px !important;
        padding: 16px 20px !important;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.02) !important;
    }
    div[data-testid="stMetric"] label {
        color: #64748B !important;
        font-size: 0.8rem !important;
        font-weight: 700 !important;
        text-transform: uppercase;
    }
    div[data-testid="stMetricValue"] {
        color: #0B2265 !important;
        font-size: 1.8rem !important;
        font-weight: 800 !important;
    }

    /* BOTONES Y SLIDERS */
    .stButton>button {
        background-color: #008037 !important;
        color: #FFFFFF !important;
        border-radius: 8px !important;
        border: none !important;
        font-weight: 700 !important;
        padding: 10px 24px !important;
    }
    .stButton>button:hover {
        background-color: #006028 !important;
    }
</style>
""", unsafe_allow_html=True)

theme_colors = ["#0B2265", "#008037", "#D97706", "#2563EB", "#059669"]

def aplicar_estilo_grafico(fig):
    fig.update_layout(
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=dict(family="Inter, sans-serif", color="#334155"),
        colorway=theme_colors,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

# =========================================================
# CARGA DE DATOS
# =========================================================
@st.cache_data
def cargar_datos(path="data.csv"):
    df = pd.read_csv(path)
    df["$Base"] = pd.to_numeric(df["$Base"], errors="coerce")
    df["$Final"] = pd.to_numeric(df["$Final"], errors="coerce")
    df["Procedencia"] = df["Procedencia"].fillna("Desconocida").str.strip()
    df["Fecha_TS"] = pd.to_datetime(df["Fecha"], dayfirst=True, errors="coerce")
    df["Hora_Entrada"] = df["Entrada"].astype(str).str.split(":").str[0].str.zfill(2)
    df["Margen_Puja"] = df["$Final"] - df["$Base"]
    df["Hubo_Puja"] = (df["$Final"] > df["$Base"]).astype(int)
    df = df.dropna(subset=["Fecha_TS"])
    return df

try:
    df_total = cargar_datos("data.csv")
except Exception:
    st.error("Error al cargar 'data.csv'. Verifica que se encuentre subido correctamente.")
    st.stop()

SEXO_LABELS = {
    "HL": "Hembra de Levante", "ML": "Macho de Levante", "VH": "Vaca Horra",
    "HV": "Hembra de Vientre", "TR": "Ternero(a)", "MC": "Macho de Ceba",
    "VI": "Vaca Industrial", "VP": "Vaca Parida", "TO": "Toro",
    "BF": "Búfala", "BH": "Búfalo", "TI": "Toro/Otro",
}

def etiqueta_sexo(codigo: str) -> str:
    return f"{codigo} · {SEXO_LABELS.get(codigo, codigo)}"

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.markdown(LOGO_COLOR_SVG, unsafe_allow_html=True)
st.sidebar.caption("Plataforma de Analítica Predictiva — Zarzal, Valle")
st.sidebar.markdown("---")

st.sidebar.subheader("🔍 Filtros de Consulta")

fecha_min, fecha_max = df_total["Fecha_TS"].min().date(), df_total["Fecha_TS"].max().date()
rango_fechas = st.sidebar.date_input(
    "Rango de fechas", value=(fecha_min, fecha_max), min_value=fecha_min, max_value=fecha_max
)

sexos_disp = sorted(df_total["Sexo"].unique())
sexos_sel = st.sidebar.multiselect(
    "Categoría (Sexo)", options=sexos_disp, default=sexos_disp,
    format_func=etiqueta_sexo,
)

procedencias_disp = sorted(df_total["Procedencia"].unique())
procedencias_sel = st.sidebar.multiselect(
    "Procedencia (deja vacío = todas)", options=procedencias_disp, default=[]
)

st.sidebar.markdown("---")
st.sidebar.caption("Suganorte S.A. — Zarzal, Valle del Cauca")

# Aplicar Filtros
if isinstance(rango_fechas, tuple) and len(rango_fechas) == 2:
    f_ini, f_fin = rango_fechas
else:
    f_ini, f_fin = fecha_min, fecha_max

mask = (
    (df_total["Fecha_TS"].dt.date >= f_ini)
    & (df_total["Fecha_TS"].dt.date <= f_fin)
    & (df_total["Sexo"].isin(sexos_sel))
)
if procedencias_sel:
    mask &= df_total["Procedencia"].isin(procedencias_sel)

df = df_total[mask].copy()

# =========================================================
# BANNER PRINCIPAL (SINTAXIS HTML CORREGIDA)
# =========================================================
st.markdown(f"""
<div class="suganorte-header-container">
    <div class="tricolor-stripe"></div>
    <div class="suganorte-banner-body">
        <div>
            <h1 class="suganorte-title">Plataforma de Analítica Predictiva</h1>
            <p class="suganorte-subtitle">Subasta Ganadera Suganorte S.A. · Histórico & Pronósticos</p>
        </div>
        <div>
            {LOGO_WHITE_SVG}
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

if df.empty:
    st.warning("No hay datos para los filtros seleccionados.")
    st.stop()

# =========================================================
# MENÚ NAVEGACIÓN
# =========================================================
selected_tab = option_menu(
    menu_title=None,
    options=["Resumen", "Exploración", "Correlación", "Predictor", "Pronóstico", "Prob. Puja", "Compradores"],
    icons=["bar-chart-fill", "search", "link-45deg", "calculator-fill", "graph-up-arrow", "diagram-3-fill", "people-fill"],
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {"padding": "6px!important", "background-color": "#FFFFFF", "border-radius": "10px", "border": "1px solid #E2E8F0"},
        "icon": {"color": "#0B2265", "font-size": "14px"},
        "nav-link": {"font-size": "13px", "text-align": "center", "margin": "2px", "color": "#475569", "font-weight": "600", "border-radius": "6px"},
        "nav-link-selected": {"background-color": "#0B2265", "color": "#FFFFFF", "font-weight": "700"},
    }
)

# ---------------------------------------------------------
# CONTENIDO DE PESTAÑAS
# ---------------------------------------------------------
if selected_tab == "Resumen":
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Lotes vendidos", f"{len(df):,}")
    c2.metric("Animales totales", f"{int(df['Cant.'].sum()):,}")
    c3.metric("Precio Prom. ($/Kg)", f"${df['$Final'].mean():,.0f}")
    c4.metric("Peso Prom. (Kg)", f"{df['P.Prom'].mean():,.1f}")
    tasa_puja = df["Hubo_Puja"].mean() * 100
    c5.metric("Lotes con Puja", f"{tasa_puja:,.1f}%")

    st.markdown("### Evolución del precio final por Kg")
    serie_diaria = df.groupby("Fecha_TS")["$Final"].mean().reset_index()
    fig = px.line(serie_diaria, x="Fecha_TS", y="$Final", markers=True)
    fig.update_layout(xaxis_title="Fecha", yaxis_title="Precio Final Promedio ($/Kg)")
    fig = aplicar_estilo_grafico(fig)
    st.plotly_chart(fig, use_container_width=True)

elif selected_tab == "Pronóstico":
    st.markdown("### Pronóstico de precio semanal (Holt-Winters)")
    df_ts = df_total.copy().set_index("Fecha_TS").sort_index()
    precio_semanal = df_ts["$Final"].resample("W").mean().ffill()

    if len(precio_semanal) >= 8:
        semanas_futuro = st.slider("Semanas a pronosticar hacia adelante", 2, 12, 4)
        modelo_full = ExponentialSmoothing(
            precio_semanal, trend="add", seasonal=None, initialization_method="estimated"
        ).fit()
        forecast_full = modelo_full.forecast(semanas_futuro)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=precio_semanal.index, y=precio_semanal.values,
                                 mode="lines+markers", name="Histórico", line=dict(color="#0B2265")))
        fig.add_trace(go.Scatter(x=forecast_full.index, y=forecast_full.values,
                                 mode="lines+markers", name="Pronóstico", line=dict(dash="dash", color="#008037")))
        fig.update_layout(xaxis_title="Semana", yaxis_title="Precio Final Promedio ($/Kg)")
        fig = aplicar_estilo_grafico(fig)
        st.plotly_chart(fig, use_container_width=True)

# FOOTER
st.markdown("---")
st.caption(
    "Proyecto de Analítica Predictiva — Suganorte S.A. (Zarzal, Valle) · "
    "Integrantes: Jeferson Balcazar Gomez, Carlos Arturo Agudelo Garcia, Milton Vanegas Delgado."
)
