"""
Dashboard de Analítica Predictiva - Subasta Ganadera Suganorte S.A. (Zarzal, Valle)
Interfaz Institucional Oficial alineada a la marca Suganorte S.A.
Código Completo Integrado
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
    df["Procedencia"] = df["Procedencia"].fillna("Desconocida").str.strip()
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
    
    # Prima porcentual de puja
    df["Prima_Puja_Pct"] = np.where(df["$Base"] > 0, ((df["$Final"] - df["$Base"]) / df["$Base"]) * 100, 0)
    
    # Clasificación por propósito según rango de peso
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

# Exportación de datos filtrados
csv_export = df_total.to_csv(index=False).encode('utf-8')
st.sidebar.download_button(
    label="📥 Descargar Dataset Filtrado (CSV)",
    data=csv_export,
    file_name="suganorte_datos_filtrados.csv",
    mime="text/csv",
)

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
# TAB 1: RESUMEN (PORTADA EJECUTIVA DE IMPACTO)
# ---------------------------------------------------------
if selected_tab == "Resumen":
    # Cálculos para Métricas Principales
    monto_total_comercializado = df["Monto_Lote"].sum()
    toneladas_totales = df["Peso_Total"].sum() / 1000
    precio_prom = df["$Final"].mean()
    prima_prom_pct = df["Prima_Puja_Pct"].mean()
    tasa_puja = df["Hubo_Puja"].mean() * 100

    # 1. KPIs FINANCIEROS Y OPERATIVOS
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    
    kpi1.metric(
        "Monto Movilizado", 
        f"${monto_total_comercializado / 1e6:,.1f} M",
        help="Valor comercial total adjudicado en subasta (Millones COP)"
    )
    kpi2.metric(
        "Tonelaje Comercializado", 
        f"{toneladas_totales:,.1f} Ton",
        help="Toneladas de peso vivo vendidas"
    )
    kpi3.metric(
        "Precio Promedio", 
        f"${precio_prom:,.0f} /Kg",
        help="Precio promedio final por kilogramo"
    )
    kpi4.metric(
        "Prima de Subasta", 
        f"+{prima_prom_pct:.1f}%",
        help="Valorización media lograda sobre el precio de salida base"
    )
    kpi5.metric(
        "Efectividad de Puja", 
        f"{tasa_puja:,.1f}%",
        help="Porcentaje de lotes que recibieron ofertas efectivas"
    )

    st.markdown("---")

    # 2. ASPECTOS CLAVE DEL PERIODO (HIGHLIGHTS)
    top_cat_vol = df["Sexo"].mode()[0] if not df.empty else "N/A"
    top_proc_vol = df["Procedencia"].mode()[0] if not df.empty else "N/A"
    precio_max = df["$Final"].max()
    lote_max_precio = df.loc[df["$Final"].idxmax()] if not df.empty else None

    st.markdown("### 🌟 Aspectos Clave del Mercado")
    
    col_h1, col_h2, col_h3, col_h4 = st.columns(4)
    
    with col_h1:
        st.markdown(f"""
        <div style="background-color: #FFFFFF; border: 1px solid #E2E8F0; border-left: 4px solid #003399; border-radius: 8px; padding: 12px 16px;">
            <p style="margin: 0; font-size: 0.75rem; color: #64748B; font-weight: 700; text-transform: uppercase;">Categoría Predominante</p>
            <h4 style="margin: 4px 0 0 0; color: #003399; font-size: 1.1rem;">{etiqueta_sexo(top_cat_vol)}</h4>
        </div>
        """, unsafe_allow_html=True)

    with col_h2:
        st.markdown(f"""
        <div style="background-color: #FFFFFF; border: 1px solid #E2E8F0; border-left: 4px solid #008037; border-radius: 8px; padding: 12px 16px;">
            <p style="margin: 0; font-size: 0.75rem; color: #64748B; font-weight: 700; text-transform: uppercase;">Principal Procedencia</p>
            <h4 style="margin: 4px 0 0 0; color: #008037; font-size: 1.1rem;">{top_proc_vol}</h4>
        </div>
        """, unsafe_allow_html=True)

    with col_h3:
        st.markdown(f"""
        <div style="background-color: #FFFFFF; border: 1px solid #E2E8F0; border-left: 4px solid #D97706; border-radius: 8px; padding: 12px 16px;">
            <p style="margin: 0; font-size: 0.75rem; color: #64748B; font-weight: 700; text-transform: uppercase;">Récord de Precio $/Kg</p>
            <h4 style="margin: 4px 0 0 0; color: #D97706; font-size: 1.1rem;">${precio_max:,.0f} /Kg</h4>
        </div>
        """, unsafe_allow_html=True)

    with col_h4:
        st.markdown(f"""
        <div style="background-color: #FFFFFF; border: 1px solid #E2E8F0; border-left: 4px solid #2563EB; border-radius: 8px; padding: 12px 16px;">
            <p style="margin: 0; font-size: 0.75rem; color: #64748B; font-weight: 700; text-transform: uppercase;">Volumen Movilizado</p>
            <h4 style="margin: 4px 0 0 0; color: #2563EB; font-size: 1.1rem;">{len(df):,} Lotes ({int(df['Cant.'].sum()):,} Cabezas)</h4>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # 3. GRÁFICO COMBINADO: TENDENCIA DIARIA DE PRECIO Y CABEZAS
    st.markdown("### 📈 Tendencia del Mercado: Precio Promedio vs. Cabezas Subastadas")
    
    df_diario = df.groupby("Fecha_TS").agg(
        Precio_Prom=("$Final", "mean"),
        Cabezas_Totales=("Cant.", "sum")
    ).reset_index()

    fig_comb = go.Figure()

    # Barras: Volumen
    fig_comb.add_trace(go.Bar(
        x=df_diario["Fecha_TS"],
        y=df_diario["Cabezas_Totales"],
        name="Cabezas Subastadas",
        marker_color="#93C5FD",
        opacity=0.6,
        yaxis="y2"
    ))

    # Línea: Precio
    fig_comb.add_trace(go.Scatter(
        x=df_diario["Fecha_TS"],
        y=df_diario["Precio_Prom"],
        name="Precio Promedio ($/Kg)",
        mode="lines+markers",
        line=dict(color="#003399", width=3),
        marker=dict(size=6, color="#008037")
    ))

    fig_comb.update_layout(
        xaxis=dict(title="Fecha de Subasta"),
        yaxis=dict(title="Precio Promedio ($/Kg)", side="left"),
        yaxis2=dict(title="Cabezas de Ganado", overlaying="y", side="right", showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(aplicar_estilo_grafico(fig_comb), use_container_width=True)

    # 4. TRÍO DE DESGLOSES COMERCIALES
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown("#### 🥩 Distribución por Propósito (Rango Peso)")
        df_rango = df["Rango_Peso"].value_counts().reset_index()
        df_rango.columns = ["Propósito", "Lotes"]
        fig_donut = px.pie(
            df_rango, 
            names="Propósito", 
            values="Lotes", 
            hole=0.4,
            color_discrete_sequence=["#003399", "#008037", "#D97706"]
        )
        fig_donut = aplicar_estilo_grafico(fig_donut)
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_b:
        st.markdown("#### 💰 Ingresos por Categoría ($ COP)")
        ingreso_sexo = df.groupby("Sexo")["Monto_Lote"].sum().reset_index()
        ingreso_sexo["Etiqueta"] = ingreso_sexo["Sexo"].map(etiqueta_sexo)
        ingreso_sexo = ingreso_sexo.sort_values("Monto_Lote", ascending=True).tail(6)

        fig_ing_sexo = px.bar(
            ingreso_sexo, 
            x="Monto_Lote", 
            y="Etiqueta", 
            orientation="h",
            text_auto=".2s",
            labels={"Monto_Lote": "COP", "Etiqueta": ""}
        )
        fig_ing_sexo = aplicar_estilo_grafico(fig_ing_sexo)
        st.plotly_chart(fig_ing_sexo, use_container_width=True)

    with col_c:
        st.markdown("#### 🏆 Top Procedencias por Monto ($ COP)")
        ingreso_proc = df.groupby("Procedencia")["Monto_Lote"].sum().reset_index().sort_values("Monto_Lote", ascending=False).head(6)
        ingreso_proc = ingreso_proc.sort_values("Monto_Lote", ascending=True)

        fig_ing_proc = px.bar(
            ingreso_proc, 
            x="Monto_Lote", 
            y="Procedencia", 
            orientation="h",
            text_auto=".2s",
            color="Monto_Lote",
            color_continuous_scale=["#93C5FD", "#003399"],
            labels={"Monto_Lote": "COP", "Procedencia": ""}
        )
        fig_ing_proc.update_layout(coloraxis_showscale=False)
        fig_ing_proc = aplicar_estilo_grafico(fig_ing_proc)
        st.plotly_chart(fig_ing_proc, use_container_width=True)

# ---------------------------------------------------------
# TAB 2: EXPLORACIÓN (VISUALIZACIÓN COMPLETA DE GRÁFICOS)
# ---------------------------------------------------------
elif selected_tab == "Exploración":
    st.markdown("### 📈 Visualización Exploratoria Comercial y Operativa")
    st.caption("Análisis gráfico interactivo sobre la estructura de precios, franjas horarias, orígenes y dinámica de puja.")

    top5 = df["Sexo"].value_counts().head(5).index
    df_top5 = df[df["Sexo"].isin(top5)].copy()
    df_top5["Etiqueta"] = df_top5["Sexo"].map(etiqueta_sexo)

    # 1. BLOQUE DE GRÁFICOS: RELACIONES DE PRECIO Y LOTE
    col_g1, col_g2 = st.columns(2)

    with col_g1:
        st.markdown("#### ⚖️ Peso vs. Precio Final por Categoría (Top 5)")
        fig_scatter = px.scatter(
            df_top5, 
            x="P.Prom", 
            y="$Final", 
            color="Etiqueta",
            opacity=0.6, 
            trendline="ols",
            labels={"P.Prom": "Peso Promedio (Kg)", "$Final": "Precio Final ($/Kg)"}
        )
        fig_scatter = aplicar_estilo_grafico(fig_scatter)
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col_g2:
        st.markdown("#### 📦 Rango de Precios por Categoría (Boxplot Top 5)")
        orden_cat = df_top5.groupby("Etiqueta")["$Final"].median().sort_values(ascending=False).index
        fig_box = px.box(
            df_top5, 
            x="Etiqueta", 
            y="$Final", 
            color="Etiqueta",
            category_orders={"Etiqueta": list(orden_cat)}
        )
        fig_box.update_layout(showlegend=False, xaxis_title="Categoría", yaxis_title="Precio Final ($/Kg)")
        fig_box = aplicar_estilo_grafico(fig_box)
        st.plotly_chart(fig_box, use_container_width=True)

    # 2. BLOQUE DE GRÁFICOS: OPERATIVO Y GEOGRÁFICO
    col_g3, col_g4 = st.columns(2)

    with col_g3:
        st.markdown("#### ⏰ Precio Promedio por Hora de Entrada")
        precio_hora = df.groupby("Hora_Entrada")["$Final"].mean().reset_index()
        fig_hora = px.bar(
            precio_hora, 
            x="Hora_Entrada", 
            y="$Final",
            text_auto=".0f",
            labels={"Hora_Entrada": "Hora de Entrada", "$Final": "Precio Promedio ($/Kg)"},
            color="$Final",
            color_continuous_scale=["#93C5FD", "#003399"]
        )
        fig_hora.update_layout(coloraxis_showscale=False)
        fig_hora = aplicar_estilo_grafico(fig_hora)
        st.plotly_chart(fig_hora, use_container_width=True)

    with col_g4:
        st.markdown("#### 🗺️ Origen del Ganado (Treemap de Procedencias)")
        proc_tree = df.groupby("Procedencia").agg(
            Lotes=("Cant.", "count"),
            Precio_Prom=("$Final", "mean")
        ).reset_index()
        
        fig_tree = px.treemap(
            proc_tree, 
            path=["Procedencia"], 
            values="Lotes",
            color="Precio_Prom",
            color_continuous_scale="Greens",
            labels={"Precio_Prom": "Precio Prom. ($/Kg)", "Lotes": "Total Lotes"}
        )
        fig_tree = aplicar_estilo_grafico(fig_tree)
        st.plotly_chart(fig_tree, use_container_width=True)

    # 3. BLOQUE DE GRÁFICOS: COMPETITIVIDAD Y TAMAÑO DE LOTE
    col_g5, col_g6 = st.columns(2)

    with col_g5:
        st.markdown("#### 🎻 Distribución del Margen de Puja ($Final - $Base)")
        fig_violin = px.violin(
            df_top5, 
            y="Margen_Puja", 
            x="Etiqueta", 
            color="Etiqueta",
            box=True, 
            points=False,
            labels={"Margen_Puja": "Margen de Puja ($)", "Etiqueta": "Categoría"}
        )
        fig_violin.update_layout(showlegend=False)
        fig_violin = aplicar_estilo_grafico(fig_violin)
        st.plotly_chart(fig_violin, use_container_width=True)

    with col_g6:
        st.markdown("#### 🫧 Cabezas por Lote vs. Precio Final")
        fig_bubble = px.scatter(
            df_top5, 
            x="Cant.", 
            y="$Final", 
            size="P.Prom", 
            color="Etiqueta",
            hover_data=["Procedencia"],
            labels={"Cant.": "Animales en Lote", "$Final": "Precio Final ($/Kg)", "P.Prom": "Peso Prom. (Kg)"}
        )
        fig_bubble = aplicar_estilo_grafico(fig_bubble)
        st.plotly_chart(fig_bubble, use_container_width=True)

    # 4. HISTOGRAMA GENERAL
    st.markdown("---")
    st.markdown("### 📊 Histograma de Distribución General de Precios")

    fig_hist = px.histogram(df, x="$Final", nbins=40)
    promedio = df["$Final"].mean()
    fig_hist.add_vline(
        x=promedio, 
        line_dash="dash", 
        line_color="#E11D48",
        annotation_text=f"Promedio: ${promedio:,.0f}"
    )
    fig_hist.update_layout(xaxis_title="Precio Final ($/Kg)", yaxis_title="Frecuencia (Lotes)")
    fig_hist = aplicar_estilo_grafico(fig_hist)
    st.plotly_chart(fig_hist, use_container_width=True)

# ---------------------------------------------------------
# TAB 3: CORRELACIÓN
# ---------------------------------------------------------
elif selected_tab == "Correlación":
    st.markdown("### 📊 Matriz de Correlación Multivariable (Heatmap)")
    
    cols_heatmap = ["$Final", "$Base", "P.Prom", "Cant.", "Peso_Total", "Margen_Puja", "Hubo_Puja", "Hora_Num"]
    labels_heatmap = ["Precio Final", "Precio Base", "Peso Prom.", "Cantidad", "Peso Total Lote", "Margen Puja", "Hubo Puja", "Hora Entrada"]
    
    corr_matrix = df[cols_heatmap].corr(numeric_only=True)

    fig_heatmap = px.imshow(
        corr_matrix,
        text_auto=".2f",
        color_continuous_scale=["#E11D48", "#F8FAFC", "#008037"],
        labels=dict(color="Correlación"),
        x=labels_heatmap,
        y=labels_heatmap
    )
    fig_heatmap.update_layout(height=580)
    fig_heatmap = aplicar_estilo_grafico(fig_heatmap)
    st.plotly_chart(fig_heatmap, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🎯 Impacto por Categoría y Variable vs. Precio Final")
    
    df_dummies = pd.get_dummies(df, columns=["Sexo"], drop_first=True)
    cols_num = ["Cant.", "P.Prom", "Peso_Total", "$Base", "$Final"] + [c for c in df_dummies.columns if c.startswith("Sexo_")]
    corr_obj = df_dummies[cols_num].corr(numeric_only=True)[["$Final"]].sort_values(by="$Final", ascending=False)
    corr_obj = corr_obj.drop(index="$Final")

    fig_bar = px.bar(
        corr_obj, 
        x="$Final", 
        y=corr_obj.index, 
        orientation="h",
        color="$Final", 
        color_continuous_scale=["#E11D48", "#F8FAFC", "#008037"],
        labels={"$Final": "Correlación con Precio Final", "index": "Variable / Categoría"}
    )
    fig_bar.update_layout(height=500)
    fig_bar = aplicar_estilo_grafico(fig_bar)
    st.plotly_chart(fig_bar, use_container_width=True)

# ---------------------------------------------------------
# TAB 4: PREDICTOR COMPARATIVO CON GRÁFICOS DIAGNÓSTICOS
# ---------------------------------------------------------
elif selected_tab == "Predictor":
    @st.cache_resource
    def entrenar_modelos(df_in: pd.DataFrame):
        d = df_in.copy()
        d["Hora_Entrada"] = "Hora_" + d["Hora_Entrada"].astype(str)

        d_model = pd.get_dummies(d, columns=["Sexo", "Procedencia", "Hora_Entrada"], drop_first=True)
        cols_sexo = [c for c in d_model.columns if c.startswith("Sexo_")]
        cols_proc = [c for c in d_model.columns if c.startswith("Procedencia_")]
        cols_hora = [c for c in d_model.columns if c.startswith("Hora_Entrada_")]
        
        columnas_x = ["P.Prom", "Cant.", "Peso_Total", "Es_Lote_Multiple", "Mes", "Dia_Semana"] + cols_sexo + cols_proc + cols_hora

        X = d_model[columnas_x]
        y = d_model["$Final"]

        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

        # 1. Regresión Lineal Múltiple (Baseline)
        modelo_lr = LinearRegression().fit(X_tr, y_tr)
        y_pred_lr = modelo_lr.predict(X_te)
        metricas_lr = {
            "R2": r2_score(y_te, y_pred_lr),
            "MAE": mean_absolute_error(y_te, y_pred_lr),
            "RMSE": np.sqrt(mean_squared_error(y_te, y_pred_lr))
        }

        # 2. Random Forest Regressor (Ensamble No Lineal)
        modelo_rf = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1).fit(X_tr, y_tr)
        y_pred_rf = modelo_rf.predict(X_te)
        metricas_rf = {
            "R2": r2_score(y_te, y_pred_rf),
            "MAE": mean_absolute_error(y_te, y_pred_rf),
            "RMSE": np.sqrt(mean_squared_error(y_te, y_pred_rf))
        }

        return {
            "modelo_lr": modelo_lr, "metricas_lr": metricas_lr,
            "modelo_rf": modelo_rf, "metricas_rf": metricas_rf,
            "columnas_x": columnas_x,
            "y_test": y_te,
            "y_pred_lr": y_pred_lr,
            "y_pred_rf": y_pred_rf,
            "sexos": sorted(d["Sexo"].unique()),
            "procedencias": sorted(d["Procedencia"].unique()),
            "horas": sorted(d["Hora_Entrada"].astype(str).unique()),
        }

    modelos = entrenar_modelos(df)
    m_lr = modelos["metricas_lr"]
    m_rf = modelos["metricas_rf"]

    st.markdown("### 📊 Tabla Comparativa de Desempeño Técnico")
    
    df_comparativa = pd.DataFrame({
        "Métrica": ["R² (Varianza explicada)", "MAE (Error Absoluto Medio)", "RMSE (Sensibilidad a Outliers)"],
        "Regresión Lineal (Baseline)": [f"{m_lr['R2']*100:.2f}%", f"${m_lr['MAE']:,.2f} /Kg", f"${m_lr['RMSE']:,.2f} /Kg"],
        "Random Forest Regressor": [f"{m_rf['R2']*100:.2f}%", f"${m_rf['MAE']:,.2f} /Kg", f"${m_rf['RMSE']:,.2f} /Kg"],
        "Diferencia / Ganancia": [
            f"+{(m_rf['R2'] - m_lr['R2'])*100:+.2f}% pts",
            f"${m_rf['MAE'] - m_lr['MAE']:,.2f} /Kg ({'Mejora' if m_rf['MAE'] < m_lr['MAE'] else 'Peor'})",
            f"${m_rf['RMSE'] - m_lr['RMSE']:,.2f} /Kg ({'Mejora' if m_rf['RMSE'] < m_lr['RMSE'] else 'Peor'})"
        ]
    })
    st.table(df_comparativa)

    col1, col2, col3 = st.columns(3)
    col1.metric("R² Random Forest", f"{m_rf['R2']*100:.1f}%", delta=f"{(m_rf['R2'] - m_lr['R2'])*100:+.1f}% pts vs LR")
    col2.metric("MAE (Error Promedio)", f"${m_rf['MAE']:,.0f}", delta=f"${m_rf['MAE'] - m_lr['MAE']:,.0f}", delta_color="inverse")
    col3.metric("RMSE (Penalización Outliers)", f"${m_rf['RMSE']:,.0f}", delta=f"${m_rf['RMSE'] - m_lr['RMSE']:,.0f}", delta_color="inverse")

    st.markdown("---")

    # --- SECCIÓN DE GRÁFICOS DIAGNÓSTICOS DE LOS MODELOS ---
    st.markdown("### 📈 Evaluador Gráfico: Regresión Lineal vs. Random Forest")
    st.caption("Comparativa de ajuste directo entre valores reales y predichos junto a la distribución del error de predicción.")

    df_eval = pd.DataFrame({
        "Real": modelos["y_test"],
        "Regresión Lineal": modelos["y_pred_lr"],
        "Random Forest": modelos["y_pred_rf"]
    })

    col_g1, col_g2 = st.columns(2)

    with col_g1:
        st.markdown("#### 🎯 Val. Reales vs. Predichos (Ajuste)")
        fig_real_pred = go.Figure()

        min_val = min(df_eval["Real"].min(), df_eval["Random Forest"].min())
        max_val = max(df_eval["Real"].max(), df_eval["Random Forest"].max())

        # Línea de referencia y = x
        fig_real_pred.add_trace(go.Scatter(
            x=[min_val, max_val], y=[min_val, max_val],
            mode='lines', name='Predicción Perfecta (y=x)',
            line=dict(color='#E11D48', dash='dash')
        ))

        # Dispersión Regresión Lineal
        fig_real_pred.add_trace(go.Scatter(
            x=df_eval["Real"], y=df_eval["Regresión Lineal"],
            mode='markers', name='Regresión Lineal',
            marker=dict(color='#2563EB', opacity=0.4, size=6)
        ))

        # Dispersión Random Forest
        fig_real_pred.add_trace(go.Scatter(
            x=df_eval["Real"], y=df_eval["Random Forest"],
            mode='markers', name='Random Forest',
            marker=dict(color='#008037', opacity=0.6, size=6)
        ))

        fig_real_pred.update_layout(
            xaxis_title="Precio Real ($/Kg)",
            yaxis_title="Precio Predicho ($/Kg)"
        )
        st.plotly_chart(aplicar_estilo_grafico(fig_real_pred), use_container_width=True)

    with col_g2:
        st.markdown("#### 📉 Distribución de Errores (Residuos)")
        df_eval["Residuo_LR"] = df_eval["Real"] - df_eval["Regresión Lineal"]
        df_eval["Residuo_RF"] = df_eval["Real"] - df_eval["Random Forest"]

        fig_res = go.Figure()
        fig_res.add_trace(go.Histogram(
            x=df_eval["Residuo_LR"], name="Errores Reg. Lineal",
            marker_color="#2563EB", opacity=0.5, nbinsx=35
        ))
        fig_res.add_trace(go.Histogram(
            x=df_eval["Residuo_RF"], name="Errores Random Forest",
            marker_color="#008037", opacity=0.6, nbinsx=35
        ))

        fig_res.update_layout(
            barmode='overlay',
            xaxis_title="Error de Predicción ($/Kg)",
            yaxis_title="Frecuencia (Lotes)"
        )
        st.plotly_chart(aplicar_estilo_grafico(fig_res), use_container_width=True)

    st.markdown("---")

    # --- CALCULADORA PREDICTIVA ---
    st.markdown("### 🧮 Calculadora Predictiva Multimodelo")

    colf1, colf2, colf3, colf4 = st.columns(4)
    with colf1:
        peso_in = st.number_input("Peso Promedio (Kg)", min_value=20, max_value=600, value=180, step=5)
    with colf2:
        cant_in = st.number_input("Cantidad de animales", min_value=1, max_value=50, value=5, step=1)
    with colf3:
        sexo_in = st.selectbox("Categoría (Sexo)", options=modelos["sexos"], format_func=etiqueta_sexo)
    with colf4:
        hora_in = st.selectbox("Hora de entrada", options=modelos["horas"])

    procedencia_in = st.selectbox("Procedencia", options=modelos["procedencias"])

    def predecir_precios(peso, cantidad, sexo, procedencia, hora):
        datos = {c: 0 for c in modelos["columnas_x"]}
        datos["P.Prom"] = peso
        datos["Cant."] = cantidad
        datos["Peso_Total"] = peso * cantidad
        datos["Es_Lote_Multiple"] = 1 if cantidad > 1 else 0
        datos["Mes"] = 8
        datos["Dia_Semana"] = 4

        if f"Sexo_{sexo}" in datos: datos[f"Sexo_{sexo}"] = 1
        if f"Procedencia_{procedencia}" in datos: datos[f"Procedencia_{procedencia}"] = 1
        if f"Hora_Entrada_{hora}" in datos: datos[f"Hora_Entrada_{hora}"] = 1

        df_p = pd.DataFrame([datos])[modelos["columnas_x"]]
        p_lr = modelos["modelo_lr"].predict(df_p)[0]
        p_rf = modelos["modelo_rf"].predict(df_p)[0]
        return p_lr, p_rf

    if st.button("💰 Calcular estimaciones comparadas", type="primary"):
        p_lr, p_rf = predecir_precios(peso_in, cant_in, sexo_in, procedencia_in, hora_in)
        
        c_res1, c_res2, c_res3 = st.columns(3)
        with c_res1:
            st.markdown("#### Regresión Lineal")
            st.write(f"**Precio/Kg:** ${p_lr:,.0f}")
            st.write(f"**Valor Lote:** ${p_lr * peso_in * cant_in:,.0f}")
        
        with c_res2:
            st.markdown("#### Random Forest (Recomendado)")
            st.write(f"**Precio/Kg:** ${p_rf:,.0f}")
            st.write(f"**Valor Lote:** ${p_rf * peso_in * cant_in:,.0f}")
        
        with c_res3:
            st.markdown("#### Discrepancia")
            dif = p_rf - p_lr
            st.write(f"**Diferencia/Kg:** ${dif:+,.0f}")
            st.caption("Random Forest se adapta mejor a patrones no lineales y combinaciones complejas de peso/categoría.")

    st.markdown("---")
    st.markdown("### 🌲 Importancia de Variables en Random Forest")
    importancias = pd.Series(
        modelos["modelo_rf"].feature_importances_, 
        index=modelos["columnas_x"]
    ).sort_values(ascending=True).tail(10)

    fig_imp = px.bar(
        x=importancias.values,
        y=importancias.index,
        orientation='h',
        labels={'x': 'Peso / Importancia Relativa', 'y': 'Variable'},
        title="Top 10 variables con mayor peso en la determinación del Precio Final"
    )
    fig_imp = aplicar_estilo_grafico(fig_imp)
    st.plotly_chart(fig_imp, use_container_width=True)

# ---------------------------------------------------------
# TAB 5: PRONÓSTICO
# ---------------------------------------------------------
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
                                 mode="lines+markers", name="Histórico", line=dict(color="#003399")))
        fig.add_trace(go.Scatter(x=forecast_full.index, y=forecast_full.values,
                                 mode="lines+markers", name="Pronóstico", line=dict(dash="dash", color="#008037")))
        fig.update_layout(xaxis_title="Semana", yaxis_title="Precio Final Promedio ($/Kg)")
        fig = aplicar_estilo_grafico(fig)
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# TAB 6: PROBABILIDAD DE PUJA (SIMULADOR Y ÁRBOL INTERACTIVO)
# ---------------------------------------------------------
elif selected_tab == "Prob. Puja":
    st.markdown("### 🎯 Simulador Interactivo y Clasificador de Probabilidad de Puja")
    st.caption("Ajusta los parámetros del modelo o los datos del lote para ver la actualización de probabilidades en tiempo real.")

    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
    with col_ctrl1:
        max_depth_sel = st.slider("Profundidad máxima del Árbol", min_value=1, max_value=8, value=3, step=1, key="tree_depth")
    with col_ctrl2:
        criterion_sel = st.selectbox("Criterio de división", options=["gini", "entropy"], format_func=lambda x: "Gini" if x == "gini" else "Entropía", key="tree_crit")
    with col_ctrl3:
        min_samples_sel = st.slider("Mínimo de muestras por nodo", min_value=2, max_value=50, value=10, step=2, key="tree_samples")

    cols_base = ["Cant.", "P.Prom", "$Base", "Procedencia", "Hubo_Puja"]
    df_tree = df[cols_base].dropna().copy()
    df_tree_encoded = pd.get_dummies(df_tree, columns=["Procedencia"], drop_first=True)

    X_tree = df_tree_encoded.drop(columns=["Hubo_Puja"])
    y_tree = df_tree_encoded["Hubo_Puja"]

    X_tr_t, X_te_t, y_tr_t, y_te_t = train_test_split(X_tree, y_tree, test_size=0.2, random_state=42)
    modelo_arbol = DecisionTreeClassifier(
        max_depth=max_depth_sel,
        criterion=criterion_sel,
        min_samples_split=min_samples_sel,
        random_state=42,
        class_weight="balanced"
    ).fit(X_tr_t, y_tr_t)

    acc_score = modelo_arbol.score(X_te_t, y_te_t) * 100
    st.info(f"💡 **Precisión (Accuracy) del Modelo:** {acc_score:.1f}% | Evaluado sobre {len(X_te_t)} lotes de prueba.")

    st.markdown("---")
    st.markdown("#### 🧪 Simulador de Lote Específico")
    col_sim1, col_sim2, col_sim3, col_sim4 = st.columns(4)

    procedencias_unicas = sorted(df["Procedencia"].dropna().unique())

    with col_sim1:
        peso_sim = st.slider("Peso Prom. (Kg)", 50, 600, 220, step=5, key="sim_peso")
    with col_sim2:
        precio_base_sim = st.slider("Precio Base ($/Kg)", 3000, 12000, 5800, step=100, key="sim_base")
    with col_sim3:
        cant_sim = st.number_input("Cantidad Animales", 1, 50, 5, step=1, key="sim_cant")
    with col_sim4:
        proc_sim = st.selectbox("Procedencia", options=procedencias_unicas, key="sim_proc")

    input_data = pd.DataFrame(0, index=[0], columns=X_tree.columns)
    input_data.loc[0, "Cant."] = cant_sim
    input_data.loc[0, "P.Prom"] = peso_sim
    input_data.loc[0, "$Base"] = precio_base_sim

    col_proc_match = f"Procedencia_{proc_sim}"
    if col_proc_match in input_data.columns:
        input_data.loc[0, col_proc_match] = 1

    probs = modelo_arbol.predict_proba(input_data)[0]
    clases = list(modelo_arbol.classes_)
    idx_puja = clases.index(1) if 1 in clases else 1
    prob_con_puja = probs[idx_puja] * 100

    col_g1, col_g2 = st.columns([1, 1])

    with col_g1:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob_con_puja,
            number={'suffix': "%", 'valueformat': ".1f"},
            title={'text': "Probabilidad de Recibir Puja"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#008037"},
                'steps': [
                    {'range': [0, 40], 'color': "#FFE4E6"},
                    {'range': [40, 70], 'color': "#FEF3C7"},
                    {'range': [70, 100], 'color': "#DCFCE7"}
                ],
                'threshold': {
                    'line': {'color': "#003399", 'width': 4},
                    'thickness': 0.75,
                    'value': prob_con_puja
                }
            }
        ))
        fig_gauge.update_layout(height=320)
        fig_gauge = aplicar_estilo_grafico(fig_gauge)
        st.plotly_chart(fig_gauge, use_container_width=True, key="gauge_plot")

    with col_g2:
        importancias = pd.DataFrame({
            "Variable": X_tree.columns,
            "Importancia": modelo_arbol.feature_importances_
        }).sort_values("Importancia", ascending=True).tail(8)

        importancias["Variable"] = importancias["Variable"].str.replace("Procedencia_", "Proc: ")

        fig_imp = px.bar(
            importancias,
            x="Importancia",
            y="Variable",
            orientation="h",
            title="Variables Más Influyentes en la Puja",
            color="Importancia",
            color_continuous_scale=["#93C5FD", "#003399"]
        )
        fig_imp.update_layout(height=320, coloraxis_showscale=False)
        fig_imp = aplicar_estilo_grafico(fig_imp)
        st.plotly_chart(fig_imp, use_container_width=True, key="imp_plot")

# ---------------------------------------------------------
# TAB 7: COMPRADORES (CLUSTERING & CARACTERIZACIÓN)
# ---------------------------------------------------------
elif selected_tab == "Compradores":
    st.markdown("### Segmentación por comportamiento de compra (K-Means)")
    st.caption("Identifica patrones de compra analizando la relación entre el peso del lote, el precio final asignado y el margen de puja obtenido.")
    
    k_sel = st.slider("Número de perfiles (K)", 2, 6, 3)
    
    cols = ["Cant.", "P.Prom", "$Base", "$Final"]
    d = df[cols].copy().dropna()
    d["Margen_Puja"] = d["$Final"] - d["$Base"]
    
    X = d[["P.Prom", "$Final", "Margen_Puja"]]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    modelo = KMeans(n_clusters=k_sel, random_state=10, n_init=10).fit(X_scaled)
    d["Perfil_ID"] = modelo.labels_

    fig = px.scatter(
        d, 
        x="P.Prom", 
        y="$Final", 
        color=d["Perfil_ID"].astype(str),
        labels={"P.Prom": "Peso Promedio (Kg)", "$Final": "Precio Final ($/Kg)", "color": "Perfil ID"},
        title="Agrupación de Lotes por Perfil de Compra"
    )
    fig = aplicar_estilo_grafico(fig)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### 📋 Caracterización y Perfilamiento de Compradores")

    resumen_clusters = d.groupby("Perfil_ID").agg(
        Lotes_Comprados=("P.Prom", "count"),
        Peso_Promedio=("P.Prom", "mean"),
        Precio_Promedio=("$Final", "mean"),
        Margen_Puja_Promedio=("Margen_Puja", "mean")
    ).reset_index()

    cols_perfiles = st.columns(min(k_sel, 3))
    
    for idx, row in resumen_clusters.iterrows():
        col_idx = idx % min(k_sel, 3)
        with cols_perfiles[col_idx]:
            p_id = int(row["Perfil_ID"])
            peso = row["Peso_Promedio"]
            precio = row["Precio_Promedio"]
            margen = row["Margen_Puja_Promedio"]
            lotes = int(row["Lotes_Comprados"])

            if peso > 350:
                tag_perfil = "🐄 Compradores de Ceba / Pesados"
            elif precio > d["$Final"].quantile(0.66):
                tag_perfil = "💎 Compradores de Alta Valoración"
            else:
                tag_perfil = "🌾 Compradores de Levante / Livianos"

            st.markdown(f"""
            <div style="background-color: #FFFFFF; border: 1px solid #E2E8F0; border-left: 5px solid #003399; border-radius: 8px; padding: 15px; margin-bottom: 15px;">
                <h4 style="margin: 0; color: #003399;">Perfil {p_id}</h4>
                <p style="font-weight: 700; color: #008037; margin: 4px 0 10px 0;">{tag_perfil}</p>
                <ul style="padding-left: 18px; margin: 0; font-size: 0.88rem; color: #334155;">
                    <li><b>Lotes adjudicados:</b> {lotes:,}</li>
                    <li><b>Peso prom. lote:</b> {peso:.1f} Kg</li>
                    <li><b>Precio prom. pagado:</b> ${precio:,.0f} /Kg</li>
                    <li><b>Margen de puja prom.:</b> ${margen:,.0f}</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

    with st.expander("📊 Ver Tabla Detallada de Métricas por Perfil"):
        st.dataframe(
            resumen_clusters.style.format({
                "Peso_Promedio": "{:.1f} Kg",
                "Precio_Promedio": "${:,.0f}",
                "Margen_Puja_Promedio": "${:,.0f}",
                "Lotes_Comprados": "{:,}"
            }),
            use_container_width=True
        )

# =========================================================
# FOOTER INSTITUCIONAL REPLICADO
# =========================================================
html_footer = """
<div class="suganorte-footer-container">
<div class="suganorte-footer-grid">
<div class="footer-col-brand">
<div class="footer-logo-title">Suganorte S.A.</div>
<div class="footer-logo-sub">Líderes en comercialización ganadera en el Suroccidente</div>
<div class="follow-text">Síguenos en</div>
<div class="footer-social-icons">
<a href="#" class="footer-social-icon" title="Instagram">📷</a>
<a href="#" class="footer-social-icon" title="Facebook">f</a>
<a href="#" class="footer-social-icon" title="YouTube">▶</a>
</div>
</div>
<div>
<div class="footer-title">Información</div>
<ul class="footer-links">
<li><span class="arrow">❯</span> Nosotros</li>
<li><span class="arrow">❯</span> Precios</li>
<li><span class="arrow">❯</span> Políticas</li>
<li><span class="arrow">❯</span> Reglamentos de la Subasta</li>
</ul>
</div>
<div>
<div class="footer-title">Servicios</div>
<ul class="footer-links">
<li><span class="arrow">❯</span> Subastas Comerciales Tradicionales</li>
<li><span class="arrow">❯</span> Subastas Adicionales</li>
<li><span class="arrow">❯</span> Remates Especializados en Fincas</li>
<li><span class="arrow">❯</span> Ventas Directas en Fincas</li>
</ul>
</div>
<div>
<div class="footer-title">Contáctenos</div>
<div class="footer-contact-info">
<p>- Km 3 Vía Zarzal - Cartago (3.25Km)</p>
<p>- <strong>WhatsApp / Celulares:</strong> 317 636 06 69<br>317 430 71 38 - 317 432 13 70</p>
<p>- <strong>Email:</strong> gerencia@suganorte.com.co</p>
<p><strong>Zarzal - Valle del Cauca</strong></p>
</div>
</div>
</div>
<div class="footer-bottom-bar">
Proyecto de Analítica Predictiva — Subasta Ganadera Suganorte S.A. | Integrantes: Jeferson Balcazar Gomez, Carlos Arturo Agudelo Garcia, Milton Vanegas Delgado.
</div>
</div>
"""

st.markdown(html_footer, unsafe_allow_html=True)
