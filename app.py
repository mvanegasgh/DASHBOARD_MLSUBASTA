"""
Dashboard de Analítica Predictiva - Subasta Ganadera Suganorte S.A. (Zarzal, Valle)
Interfaz Institucional Oficial alineada a la marca Suganorte S.A.
Código Completo Integrado y Corregido
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing

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
# ESTILOS CSS INSTITUCIONALES UNIFICADOS
# =========================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@700;800;900&family=Inter:wght@400;500;600;700&display=swap');
    @import url('https://fonts.googleapis.com/icon?family=Material+Icons|Material+Icons+Outlined|Material+Symbols+Outlined');

    /* CORRECCIÓN ICONO COLLAPSE DE STREAMLIT */
    [data-testid="stSidebarCollapseButton"] *,
    [data-testid="stSidebarCollapseButton"] button span,
    [data-testid="stSidebarCollapseButton"] span,
    i.material-icons,
    .material-icons,
    .material-symbols-outlined {
        font-family: 'Material Symbols Outlined', 'Material Icons', sans-serif !important;
        font-weight: normal !important;
        font-style: normal !important;
        text-transform: none !important;
        letter-spacing: normal !important;
        word-wrap: normal !important;
        white-space: nowrap !important;
        direction: ltr !important;
    }

    /* Fondo general */
    html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"], [data-testid="stMain"] {
        background-color: #F8FAFC !important;
        color: #0F172A !important;
        font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
    }

    /* LOGO SIDEBAR HTML */
    .sidebar-brand {
        display: flex;
        align-items: center;
        justify-content: flex-start;
        gap: 8px;
        padding: 8px 0;
    }
    .brand-suganorte {
        font-family: 'Montserrat', sans-serif;
        font-weight: 900;
        font-size: 1.65rem;
        color: #003399;
        letter-spacing: -0.8px;
        line-height: 1;
    }
    .brand-sa {
        font-family: 'Montserrat', sans-serif;
        font-weight: 800;
        font-size: 1.1rem;
        color: #003399;
        position: relative;
    }
    .brand-sa::after {
        content: "🔨";
        font-size: 0.85rem;
        position: absolute;
        top: -6px;
        right: -14px;
        transform: rotate(20deg);
    }
    .brand-stripe {
        height: 4px;
        width: 100%;
        max-width: 180px;
        background: linear-gradient(90deg, #FFD100 0% 40%, #008037 40% 75%, #E11D48 75% 100%);
        border-radius: 2px;
        margin-top: 4px;
        margin-bottom: 12px;
    }

    /* BANNER HEADER HTML */
    .suganorte-header-container {
        background: #003399;
        border-radius: 12px;
        overflow: hidden;
        margin-bottom: 24px;
        box-shadow: 0 4px 15px rgba(0, 51, 153, 0.2);
    }
    .tricolor-stripe {
        height: 6px;
        background: linear-gradient(90deg, #FFD100 0% 33%, #008037 33% 66%, #E11D48 66% 100%);
        width: 100%;
    }
    .suganorte-banner-body {
        padding: 24px 32px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 20px;
    }
    .suganorte-title {
        color: #FFFFFF !important;
        margin: 0 !important;
        font-size: 1.75rem !important;
        font-weight: 800 !important;
        letter-spacing: -0.02em;
    }
    .suganorte-subtitle {
        color: #FFD100 !important;
        margin-top: 6px !important;
        margin-bottom: 0 !important;
        font-size: 0.95rem !important;
        font-weight: 600 !important;
    }
    .banner-brand {
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .banner-suganorte {
        font-family: 'Montserrat', sans-serif;
        font-weight: 900;
        font-size: 1.8rem;
        color: #FFFFFF;
        letter-spacing: -0.8px;
    }
    .banner-sa {
        font-family: 'Montserrat', sans-serif;
        font-weight: 800;
        font-size: 1.2rem;
        color: #FFFFFF;
        position: relative;
    }
    .banner-sa::after {
        content: "🔨";
        font-size: 0.9rem;
        position: absolute;
        top: -6px;
        right: -14px;
        transform: rotate(20deg);
    }

    /* BARRA LATERAL */
    section[data-testid="stSidebar"], [data-testid="stSidebarContent"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E2E8F0 !important;
    }

    /* COMPONENTES DE ENTRADA */
    div[data-baseweb="select"] > div, 
    div[data-baseweb="input"] > div,
    input, 
    div[role="listbox"] {
        background-color: #FFFFFF !important;
        border-color: #CBD5E1 !important;
        color: #0F172A !important;
        border-radius: 8px !important;
    }

    /* TAGS MULTISELECT EN AZUL INSTITUCIONAL */
    span[data-baseweb="tag"] {
        background-color: #003399 !important;
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
        color: #003399 !important;
        font-size: 1.8rem !important;
        font-weight: 800 !important;
    }

    /* BOTONES */
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

    /* ESTILOS DEL FOOTER OFICIAL SUGANORTE S.A. */
    .suganorte-footer-container {
        background-color: #003399;
        border-top: 5px solid #008037;
        color: #FFFFFF;
        padding: 40px 30px 20px 30px;
        border-radius: 12px;
        margin-top: 50px;
        font-family: 'Inter', sans-serif;
    }
    .suganorte-footer-grid {
        display: grid;
        grid-template-columns: 1.2fr 1fr 1.2fr 1.5fr;
        gap: 30px;
    }
    .footer-col-brand .footer-logo-title {
        font-family: 'Montserrat', sans-serif;
        font-size: 1.8rem;
        font-weight: 900;
        color: #FFFFFF;
        margin-bottom: 2px;
    }
    .footer-col-brand .footer-logo-sub {
        font-size: 0.72rem;
        color: #E2E8F0;
        margin-bottom: 20px;
        letter-spacing: 0.5px;
    }
    .footer-col-brand .follow-text {
        font-size: 0.95rem;
        font-weight: 600;
        margin-bottom: 12px;
    }
    .footer-social-icons {
        display: flex;
        gap: 12px;
    }
    .footer-social-icon {
        width: 38px;
        height: 38px;
        background-color: #FFFFFF;
        color: #003399;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        font-size: 1.1rem;
        text-decoration: none;
    }
    .footer-title {
        font-size: 1.25rem;
        font-weight: 700;
        margin-bottom: 18px;
        color: #FFFFFF;
    }
    .footer-links {
        list-style: none;
        padding: 0;
        margin: 0;
    }
    .footer-links li {
        margin-bottom: 10px;
        font-size: 0.9rem;
        color: #F1F5F9;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .footer-links li span.arrow {
        font-size: 0.8rem;
        opacity: 0.8;
    }
    .footer-contact-info {
        font-size: 0.9rem;
        line-height: 1.6;
        color: #F1F5F9;
    }
    .footer-contact-info p {
        margin-bottom: 8px;
    }
    .footer-contact-info strong {
        color: #FFFFFF;
    }
    .footer-bottom-bar {
        border-top: 1px solid rgba(255, 255, 255, 0.15);
        margin-top: 30px;
        padding-top: 15px;
        text-align: center;
        font-size: 0.8rem;
        color: #94A3B8;
    }
</style>
""", unsafe_allow_html=True)

theme_colors = ["#003399", "#008037", "#D97706", "#2563EB", "#059669"]

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
# CARGA DE DATOS E INGENIERÍA DE CARACTERÍSTICAS
# =========================================================
@st.cache_data
def cargar_datos(path="data.csv"):
    df = pd.read_csv(path)
    df["$Base"] = pd.to_numeric(df["$Base"], errors="coerce")
    df["$Final"] = pd.to_numeric(df["$Final"], errors="coerce")
    df["P.Prom"] = pd.to_numeric(df["P.Prom"], errors="coerce")
    df["Cant."] = pd.to_numeric(df["Cant."], errors="coerce")
    
    # Homogeneización robusta de procedencias
    def limpiar_procedencia(x):
        if pd.isna(x): return "DESCONOCIDA"
        x = str(x).strip().upper()
        x = x.replace("Á", "A").replace("É", "E").replace("Í", "I").replace("Ó", "O").replace("Ú", "U")
        mapeo = {
            'BAGALAGRANDE': 'BUGALAGRANDE', 'BUGALAGRENDE': 'BUGALAGRANDE',
            'DOVIO': 'EL DOVIO', 'ROLDANILLA': 'ROLDANILLO', 'ROLDANLLO': 'ROLDANILLO',
            'JAMUMDI': 'JAMUNDI', 'LAUNION': 'LA UNION', 'L AUNION': 'LA UNION',
            'CAICADONIA': 'CAICEDONIA', 'EL AGULA': 'EL AGUILA', 'BOLVIAR': 'BOLIVAR',
            'RICAUTE': 'RICAURTE', 'RIO FRIA': 'RIO FRIO', 'RIOFRIO': 'RIO FRIO',
            'MANIZALEZ': 'MANIZALES', 'MARCELLA': 'MARSELLA', 'PLANETARICA': 'PLANETA RICA',
            'QIMBAYA': 'QUIMBAYA', 'QUEBRADANUEVA': 'QUEBRADA NUEVA',
            'QUEBRADAGRANDE': 'QUEBRADA GRANDE', 'TRUJILLLO': 'TRUJILLO',
            'L.A V ICTORIA': 'LA VICTORIA', 'LAHERRADURA': 'LA HERRADURA',
            'PAILARRIBA': 'PAILA ARRIBA', 'PAILA-ARRIBA': 'PAILA ARRIBA',
            'PAILA, ARRIBA': 'PAILA ARRIBA', 'PAILA  ARRIBA': 'PAILA ARRIBA',
            'SANTADER DE QUILIC': 'SANTANDER DE QUILICHAO',
            'LA MAGDALENA BUGA': 'BUGA', 'LA CHINA SEVILLA': 'SEVILLA'
        }
        if x in mapeo: return mapeo[x]
        if "SAN VICENTE" in x and "FERRER" not in x: return 'SAN VICENTE DEL CAGUAN'
        if "SANTANDER DE QUILI" in x: return 'SANTANDER DE QUILICHAO'
        if "PUERTO RICO" in x: return 'PUERTO RICO'
        if "GARRAP" in x or "CANON" in x or "CAÑON" in x or "AÑON" in x: return 'CAÑON DE GARRAPATAS'
        return x

    df["Procedencia"] = df["Procedencia"].apply(limpiar_procedencia)
    df["Fecha_TS"] = pd.to_datetime(df["Fecha"], dayfirst=True, errors="coerce")
    df["Hora_Num"] = pd.to_numeric(df["Entrada"].astype(str).str.split(":").str[0], errors="coerce")
    df["Hora_Entrada"] = df["Entrada"].astype(str).str.split(":").str[0].str.zfill(2)
    df["Margen_Puja"] = df["$Final"] - df["$Base"]
    df["Hubo_Puja"] = (df["$Final"] > df["$Base"]).astype(int)

    # --- INGENIERÍA DE CARACTERÍSTICAS DERIVADAS ---
    df["Peso_Total"] = df["P.Prom"] * df["Cant."]
    df["Es_Lote_Multiple"] = (df["Cant."] > 1).astype(int)
    df["Mes"] = df["Fecha_TS"].dt.month
    df["Dia_Semana"] = df["Fecha_TS"].dt.dayofweek
    df["Monto_Lote"] = df["$Final"] * df["P.Prom"] * df["Cant."]
    df["Prima_Puja_Pct"] = np.where(df["$Base"] > 0, ((df["$Final"] - df["$Base"]) / df["$Base"]) * 100, 0)
    
    df["Rango_Peso"] = pd.cut(
        df["P.Prom"], 
        bins=[0, 200, 350, 1000], 
        labels=["Levante (<200 Kg)", "Desarrollo (200-350 Kg)", "Ceba (>350 Kg)"]
    )

    df = df.dropna(subset=["Fecha_TS", "$Final", "P.Prom"])
    return df

try:
    df_total = cargar_datos("data.csv")
except Exception:
    st.error("Error al cargar 'data.csv'. Verifica que se encuentre en la raíz del proyecto.")
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
st.sidebar.markdown("""
<div>
    <div class="sidebar-brand">
        <span class="brand-suganorte">Suganorte</span>
        <span class="brand-sa">S.A.</span>
    </div>
    <div class="brand-stripe"></div>
</div>
""", unsafe_allow_html=True)

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

csv_export = df_total.to_csv(index=False).encode('utf-8')
st.sidebar.download_button(
    label="📥 Descargar Dataset Filtrado (CSV)",
    data=csv_export,
    file_name="suganorte_datos_filtrados.csv",
    mime="text/csv",
)

st.sidebar.caption("Suganorte S.A. — Zarzal, Valle del Cauca")

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
# BANNER PRINCIPAL
# =========================================================
st.markdown("""
<div class="suganorte-header-container">
    <div class="tricolor-stripe"></div>
    <div class="suganorte-banner-body">
        <div>
            <h1 class="suganorte-title">Plataforma de Analítica Predictiva</h1>
            <p class="suganorte-subtitle">Subasta Ganadera Suganorte S.A. · Histórico & Pronósticos</p>
        </div>
        <div class="banner-brand">
            <span class="banner-suganorte">Suganorte</span>
            <span class="banner-sa">S.A.</span>
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
        "icon": {"color": "#003399", "font-size": "14px"},
        "nav-link": {"font-size": "13px", "text-align": "center", "margin": "2px", "color": "#475569", "font-weight": "600", "border-radius": "6px"},
        "nav-link-selected": {"background-color": "#003399", "color": "#FFFFFF", "font-weight": "700"},
    }
)

# ---------------------------------------------------------
# TAB 1: RESUMEN
# ---------------------------------------------------------
if selected_tab == "Resumen":
    monto_total_comercializado = df["Monto_Lote"].sum()
    toneladas_totales = df["Peso_Total"].sum() / 1000
    precio_prom = df["$Final"].mean()
    prima_prom_pct = df["Prima_Puja_Pct"].mean()
    tasa_puja = df["Hubo_Puja"].mean() * 100

    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    kpi1.metric("Monto Movilizado", f"${monto_total_comercializado / 1e6:,.1f} M")
    kpi2.metric("Tonelaje Comercializado", f"{toneladas_totales:,.1f} Ton")
    kpi3.metric("Precio Promedio", f"${precio_prom:,.0f} /Kg")
    kpi4.metric("Prima de Subasta", f"+{prima_prom_pct:.1f}%")
    kpi5.metric("Efectividad de Puja", f"{tasa_puja:,.1f}%")

    st.markdown("---")
    top_cat_vol = df["Sexo"].mode()[0] if not df.empty else "N/A"
    top_proc_vol = df["Procedencia"].mode()[0] if not df.empty else "N/A"
    precio_max = df["$Final"].max()

    st.markdown("### 🌟 Aspectos Clave del Mercado")
    col_h1, col_h2, col_h3, col_h4 = st.columns(4)
    col_h1.markdown(f"""<div style="background:#FFF;border:1px solid #E2E8F0;border-left:4px solid #003399;border-radius:8px;padding:12px 16px;"><p style="margin:0;font-size:0.75rem;color:#64748B;font-weight:700;">CATEGORÍA</p><h4 style="margin:4px 0 0 0;color:#003399;font-size:1.1rem;">{etiqueta_sexo(top_cat_vol)}</h4></div>""", unsafe_allow_html=True)
    col_h2.markdown(f"""<div style="background:#FFF;border:1px solid #E2E8F0;border-left:4px solid #008037;border-radius:8px;padding:12px 16px;"><p style="margin:0;font-size:0.75rem;color:#64748B;font-weight:700;">PROCEDENCIA</p><h4 style="margin:4px 0 0 0;color:#008037;font-size:1.1rem;">{top_proc_vol}</h4></div>""", unsafe_allow_html=True)
    col_h3.markdown(f"""<div style="background:#FFF;border:1px solid #E2E8F0;border-left:4px solid #D97706;border-radius:8px;padding:12px 16px;"><p style="margin:0;font-size:0.75rem;color:#64748B;font-weight:700;">RÉCORD PRECIO</p><h4 style="margin:4px 0 0 0;color:#D97706;font-size:1.1rem;">${precio_max:,.0f} /Kg</h4></div>""", unsafe_allow_html=True)
    col_h4.markdown(f"""<div style="background:#FFF;border:1px solid #E2E8F0;border-left:4px solid #2563EB;border-radius:8px;padding:12px 16px;"><p style="margin:0;font-size:0.75rem;color:#64748B;font-weight:700;">LOTES</p><h4 style="margin:4px 0 0 0;color:#2563EB;font-size:1.1rem;">{len(df):,} Lotes</h4></div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📈 Tendencia del Mercado: Precio Promedio vs. Cabezas Subastadas")
    df_diario = df.groupby("Fecha_TS").agg(Precio_Prom=("$Final", "mean"), Cabezas_Totales=("Cant.", "sum")).reset_index()
    fig_comb = go.Figure()
    fig_comb.add_trace(go.Bar(x=df_diario["Fecha_TS"], y=df_diario["Cabezas_Totales"], name="Cabezas", marker_color="#93C5FD", opacity=0.6, yaxis="y2"))
    fig_comb.add_trace(go.Scatter(x=df_diario["Fecha_TS"], y=df_diario["Precio_Prom"], name="Precio Prom. ($/Kg)", mode="lines+markers", line=dict(color="#003399", width=3)))
    fig_comb.update_layout(yaxis=dict(title="Precio Promedio ($/Kg)", side="left"), yaxis2=dict(title="Cabezas", overlaying="y", side="right", showgrid=False), legend=dict(orientation="h", y=1.02, x=1, xanchor="right"))
    st.plotly_chart(aplicar_estilo_grafico(fig_comb), use_container_width=True)

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("#### 🥩 Por Propósito")
        df_rango = df["Rango_Peso"].value_counts().reset_index()
        df_rango.columns = ["Propósito", "Lotes"]
        fig_donut = px.pie(df_rango, names="Propósito", values="Lotes", hole=0.4, color_discrete_sequence=["#003399", "#008037", "#D97706"])
        st.plotly_chart(aplicar_estilo_grafico(fig_donut), use_container_width=True)
    with col_b:
        st.markdown("#### 💰 Ingresos por Categoría")
        ingreso_sexo = df.groupby("Sexo")["Monto_Lote"].sum().reset_index()
        ingreso_sexo["Etiqueta"] = ingreso_sexo["Sexo"].map(etiqueta_sexo)
        ingreso_sexo = ingreso_sexo.sort_values("Monto_Lote", ascending=True).tail(6)
        fig_ing_sexo = px.bar(ingreso_sexo, x="Monto_Lote", y="Etiqueta", orientation="h", text_auto=".2s")
        st.plotly_chart(aplicar_estilo_grafico(fig_ing_sexo), use_container_width=True)
    with col_c:
        st.markdown("#### 🏆 Top Procedencias ($)")
        ingreso_proc = df.groupby("Procedencia")["Monto_Lote"].sum().reset_index().sort_values("Monto_Lote", ascending=False).head(6).sort_values("Monto_Lote", ascending=True)
        fig_ing_proc = px.bar(ingreso_proc, x="Monto_Lote", y="Procedencia", orientation="h", text_auto=".2s", color="Monto_Lote", color_continuous_scale=["#93C5FD", "#003399"])
        fig_ing_proc.update_layout(coloraxis_showscale=False)
        st.plotly_chart(aplicar_estilo_grafico(fig_ing_proc), use_container_width=True)

# ---------------------------------------------------------
# TAB 2: EXPLORACIÓN
# ---------------------------------------------------------
elif selected_tab == "Exploración":
    st.markdown("### 📈 Visualización Exploratoria Comercial y Operativa")
    top5 = df["Sexo"].value_counts().head(5).index
    df_top5 = df[df["Sexo"].isin(top5)].copy()
    df_top5["Etiqueta"] = df_top5["Sexo"].map(etiqueta_sexo)

    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.markdown("#### ⚖️ Peso vs. Precio Final")
        fig_scatter = px.scatter(df_top5, x="P.Prom", y="$Final", color="Etiqueta", opacity=0.6, trendline="ols")
        st.plotly_chart(aplicar_estilo_grafico(fig_scatter), use_container_width=True)
    with col_g2:
        st.markdown("#### 📦 Rango de Precios por Categoría")
        fig_box = px.box(df_top5, x="Etiqueta", y="$Final", color="Etiqueta")
        fig_box.update_layout(showlegend=False)
        st.plotly_chart(aplicar_estilo_grafico(fig_box), use_container_width=True)

# ---------------------------------------------------------
# TAB 3: CORRELACIÓN
# ---------------------------------------------------------
elif selected_tab == "Correlación":
    st.markdown("### 📊 Matriz de Correlación Multivariable")
    cols_heatmap = ["$Final", "$Base", "P.Prom", "Cant.", "Peso_Total", "Margen_Puja", "Hubo_Puja", "Hora_Num"]
    corr_matrix = df[cols_heatmap].corr(numeric_only=True)
    fig_heatmap = px.imshow(corr_matrix, text_auto=".2f", color_continuous_scale=["#E11D48", "#F8FAFC", "#008037"])
    st.plotly_chart(aplicar_estilo_grafico(fig_heatmap), use_container_width=True)

# ---------------------------------------------------------
# TAB 4: PREDICTOR (CORREGIDO PARA EVITAR ERROR DE DTYPES)
# ---------------------------------------------------------
elif selected_tab == "Predictor":
    @st.cache_resource
    def entrenar_modelos(df_in: pd.DataFrame):
        d = df_in.copy()
        d["Hora_Entrada"] = "Hora_" + d["Hora_Entrada"].astype(str)
        y = pd.to_numeric(d["$Final"], errors="coerce")

        # 1. Regresión Lineal Simple
        X_simple = d[["P.Prom"]].astype(float)
        X_tr_s, X_te_s, y_tr_s, y_te_s = train_test_split(X_simple, y, test_size=0.2, random_state=42)
        modelo_simple = LinearRegression().fit(X_tr_s, y_tr_s)
        y_pred_s = modelo_simple.predict(X_te_s)
        metricas_s = {
            "R2": r2_score(y_te_s, y_pred_s),
            "MAE": mean_absolute_error(y_te_s, y_pred_s),
            "RMSE": np.sqrt(mean_squared_error(y_te_s, y_pred_s))
        }

        # 2. Variables Múltiples con conversión estricta a float
        d_model = pd.get_dummies(d, columns=["Sexo", "Procedencia", "Hora_Entrada"], drop_first=True)
        
        cols_sexo = [c for c in d_model.columns if c.startswith("Sexo_")]
        cols_proc = [c for c in d_model.columns if c.startswith("Procedencia_")]
        cols_hora = [c for c in d_model.columns if c.startswith("Hora_Entrada_")]
        
        columnas_x = ["P.Prom", "Cant.", "Peso_Total", "Es_Lote_Multiple", "Mes", "Dia_Semana"] + cols_sexo + cols_proc + cols_hora

        X_multi = d_model[columnas_x].astype(float)
        X_tr_m, X_te_m, y_tr_m, y_te_m = train_test_split(X_multi, y, test_size=0.2, random_state=42)

        modelo_lr = LinearRegression().fit(X_tr_m, y_tr_m)
        y_pred_lr = modelo_lr.predict(X_te_m)
        metricas_lr = {
            "R2": r2_score(y_te_m, y_pred_lr),
            "MAE": mean_absolute_error(y_te_m, y_pred_lr),
            "RMSE": np.sqrt(mean_squared_error(y_te_m, y_pred_lr))
        }

        modelo_rf = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1).fit(X_tr_m, y_tr_m)
        y_pred_rf = modelo_rf.predict(X_te_m)
        metricas_rf = {
            "R2": r2_score(y_te_m, y_pred_rf),
            "MAE": mean_absolute_error(y_te_m, y_pred_rf),
            "RMSE": np.sqrt(mean_squared_error(y_te_m, y_pred_rf))
        }

        return {
            "modelo_simple": modelo_simple, "metricas_s": metricas_s,
            "modelo_lr": modelo_lr, "metricas_lr": metricas_lr,
            "modelo_rf": modelo_rf, "metricas_rf": metricas_rf,
            "columnas_x": columnas_x,
            "y_test": y_te_m,
            "y_pred_s": y_pred_s,
            "y_pred_lr": y_pred_lr,
            "y_pred_rf": y_pred_rf,
            "sexos": sorted(d["Sexo"].unique()),
            "procedencias": sorted(d["Procedencia"].unique()),
            "horas": sorted(d["Hora_Entrada"].astype(str).unique()),
        }

    modelos = entrenar_modelos(df)
    m_s = modelos["metricas_s"]
    m_lr = modelos["metricas_lr"]
    m_rf = modelos["metricas_rf"]

    st.markdown("### 📊 Tabla Comparativa de Modelos Predictivos")
    df_comparativa = pd.DataFrame({
        "Métrica": ["R² (Varianza explicada)", "MAE (Error Absoluto Medio)", "RMSE (Sensibilidad a Outliers)"],
        "RL Simple (Solo Peso)": [f"{m_s['R2']*100:.2f}%", f"${m_s['MAE']:,.2f} /Kg", f"${m_s['RMSE']:,.2f} /Kg"],
        "RL Múltiple (Todas Vars)": [f"{m_lr['R2']*100:.2f}%", f"${m_lr['MAE']:,.2f} /Kg", f"${m_lr['RMSE']:,.2f} /Kg"],
        "Random Forest Regressor": [f"{m_rf['R2']*100:.2f}%", f"${m_rf['MAE']:,.2f} /Kg", f"${m_rf['RMSE']:,.2f} /Kg"]
    })
    st.table(df_comparativa)

    st.markdown("---")
    st.markdown("### 🧮 Calculadora Predictiva Trimodelo")
    colf1, colf2, colf3, colf4 = st.columns(4)
    peso_in = colf1.number_input("Peso Promedio (Kg)", min_value=20, max_value=600, value=180, step=5)
    cant_in = colf2.number_input("Cantidad", min_value=1, max_value=50, value=5, step=1)
    sexo_in = colf3.selectbox("Categoría", options=modelos["sexos"], format_func=etiqueta_sexo)
    hora_in = colf4.selectbox("Hora", options=modelos["horas"])
    procedencia_in = st.selectbox("Procedencia", options=modelos["procedencias"])

    if st.button("💰 Calcular estimaciones", type="primary"):
        p_s = modelos["modelo_simple"].predict(np.array([[peso_in]]))[0]
        
        datos = {c: 0.0 for c in modelos["columnas_x"]}
        datos["P.Prom"] = float(peso_in)
        datos["Cant."] = float(cant_in)
        datos["Peso_Total"] = float(peso_in * cant_in)
        datos["Es_Lote_Multiple"] = 1.0 if cant_in > 1 else 0.0
        datos["Mes"] = 8.0
        datos["Dia_Semana"] = 4.0

        if f"Sexo_{sexo_in}" in datos: datos[f"Sexo_{sexo_in}"] = 1.0
        if f"Procedencia_{procedencia_in}" in datos: datos[f"Procedencia_{procedencia_in}"] = 1.0
        if f"Hora_Entrada_{hora_in}" in datos: datos[f"Hora_Entrada_{hora_in}"] = 1.0

        df_p = pd.DataFrame([datos])[modelos["columnas_x"]].astype(float)
        p_lr = modelos["modelo_lr"].predict(df_p)[0]
        p_rf = modelos["modelo_rf"].predict(df_p)[0]

        c_res1, c_res2, c_res3 = st.columns(3)
        c_res1.metric("RL Simple", f"${p_s:,.0f} /Kg")
        c_res2.metric("RL Múltiple", f"${p_lr:,.0f} /Kg")
        c_res3.metric("Random Forest", f"${p_rf:,.0f} /Kg")

# ---------------------------------------------------------
# TAB 5: PRONÓSTICO
# ---------------------------------------------------------
elif selected_tab == "Pronóstico":
    st.markdown("### Pronóstico de precio semanal (Holt-Winters)")
    df_ts = df_total.copy().set_index("Fecha_TS").sort_index()
    precio_semanal = df_ts["$Final"].resample("W").mean().ffill()
    if len(precio_semanal) >= 8:
        semanas_futuro = st.slider("Semanas a pronosticar", 2, 12, 4)
        modelo_full = ExponentialSmoothing(precio_semanal, trend="add", seasonal=None, initialization_method="estimated").fit()
        forecast_full = modelo_full.forecast(semanas_futuro)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=precio_semanal.index, y=precio_semanal.values, mode="lines+markers", name="Histórico"))
        fig.add_trace(go.Scatter(x=forecast_full.index, y=forecast_full.values, mode="lines+markers", name="Pronóstico", line=dict(dash="dash")))
        st.plotly_chart(aplicar_estilo_grafico(fig), use_container_width=True)

# ---------------------------------------------------------
# TAB 6: PROBABILIDAD DE PUJA
# ---------------------------------------------------------
elif selected_tab == "Prob. Puja":
    st.markdown("### 🎯 Simulador de Probabilidad de Puja")
    cols_base = ["Cant.", "P.Prom", "$Base", "Procedencia", "Hubo_Puja"]
    df_tree = df[cols_base].dropna().copy()
    df_tree_encoded = pd.get_dummies(df_tree, columns=["Procedencia"], drop_first=True).astype(float)
    X_tree = df_tree_encoded.drop(columns=["Hubo_Puja"])
    y_tree = df_tree_encoded["Hubo_Puja"]
    
    modelo_arbol = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X_tree, y_tree)
    peso_sim = st.slider("Peso Prom. (Kg)", 50, 600, 220)
    base_sim = st.slider("Precio Base", 3000, 12000, 5800)
    
    input_data = pd.DataFrame(0.0, index=[0], columns=X_tree.columns)
    input_data.loc[0, "P.Prom"] = float(peso_sim)
    input_data.loc[0, "$Base"] = float(base_sim)
    input_data.loc[0, "Cant."] = 1.0
    
    prob = modelo_arbol.predict_proba(input_data)[0][1] * 100
    st.metric("Probabilidad estimada de Puja", f"{prob:.1f}%")

# ---------------------------------------------------------
# TAB 7: COMPRADORES
# ---------------------------------------------------------
elif selected_tab == "Compradores":
    st.markdown("### Segmentación por comportamiento de compra (K-Means)")
    k_sel = st.slider("Número de perfiles (K)", 2, 6, 3)
    cols = ["Cant.", "P.Prom", "$Base", "$Final"]
    d = df[cols].astype(float).copy().dropna()
    d["Margen_Puja"] = d["$Final"] - d["$Base"]
    X_scaled = StandardScaler().fit_transform(d[["P.Prom", "$Final", "Margen_Puja"]])
    d["Perfil_ID"] = KMeans(n_clusters=k_sel, random_state=10, n_init=10).fit_predict(X_scaled)
    fig = px.scatter(d, x="P.Prom", y="$Final", color=d["Perfil_ID"].astype(str))
    st.plotly_chart(aplicar_estilo_grafico(fig), use_container_width=True)

# =========================================================
# FOOTER INSTITUCIONAL
# =========================================================
st.markdown("""
<div class="suganorte-footer-container">
<div class="suganorte-footer-grid">
<div class="footer-col-brand">
<div class="footer-logo-title">Suganorte S.A.</div>
<div class="footer-logo-sub">Líderes en comercialización ganadera en el Suroccidente</div>
</div>
</div>
<div class="footer-bottom-bar">
Proyecto de Analítica Predictiva — Subasta Ganadera Suganorte S.A.
</div>
</div>
""", unsafe_allow_html=True)
