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
# ESTILOS CSS INSTITUCIONALES Y FOOTER OFICIAL
# =========================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@700;800;900&family=Inter:wght@400;500;600;700&display=swap');

    /* Fondo general */
    html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"], [data-testid="stMain"] {
        background-color: #F8FAFC !important;
        color: #0F172A !important;
        font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
    }

    /* Ajuste para iconos nativos */
    [data-testid="stSidebarCollapseButton"] button span,
    [data-testid="stSidebarCollapseButton"] span {
        font-family: "Source Sans Pro", sans-serif, "Material Icons" !important;
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
        color: #0B2265;
        letter-spacing: -0.8px;
        line-height: 1;
    }
    .brand-sa {
        font-family: 'Montserrat', sans-serif;
        font-weight: 800;
        font-size: 1.1rem;
        color: #0B2265;
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
    df["Hora_Num"] = pd.to_numeric(df["Entrada"].astype(str).str.split(":").str[0], errors="coerce")
    df["Hora_Entrada"] = df["Entrada"].astype(str).str.split(":").str[0].str.zfill(2)
    df["Margen_Puja"] = df["$Final"] - df["$Base"]
    df["Hubo_Puja"] = (df["$Final"] > df["$Base"]).astype(int)
    df = df.dropna(subset=["Fecha_TS"])
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
        "icon": {"color": "#0B2265", "font-size": "14px"},
        "nav-link": {"font-size": "13px", "text-align": "center", "margin": "2px", "color": "#475569", "font-weight": "600", "border-radius": "6px"},
        "nav-link-selected": {"background-color": "#0B2265", "color": "#FFFFFF", "font-weight": "700"},
    }
)

# ---------------------------------------------------------
# TAB 1: RESUMEN
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

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### Volumen por Categoría (Sexo)")
        vol_sexo = df["Sexo"].value_counts().reset_index()
        vol_sexo.columns = ["Sexo", "Lotes"]
        vol_sexo["Etiqueta"] = vol_sexo["Sexo"].map(etiqueta_sexo)
        fig2 = px.bar(vol_sexo, x="Lotes", y="Etiqueta", orientation="h")
        fig2.update_layout(yaxis_title="")
        fig2 = aplicar_estilo_grafico(fig2)
        st.plotly_chart(fig2, use_container_width=True)
    with col_b:
        st.markdown("### Top 10 Procedencias por volumen")
        vol_proc = df["Procedencia"].value_counts().head(10).reset_index()
        vol_proc.columns = ["Procedencia", "Lotes"]
        fig3 = px.bar(vol_proc, x="Lotes", y="Procedencia", orientation="h")
        fig3.update_layout(yaxis_title="")
        fig3 = aplicar_estilo_grafico(fig3)
        st.plotly_chart(fig3, use_container_width=True)

# ---------------------------------------------------------
# TAB 2: EXPLORACIÓN
# ---------------------------------------------------------
elif selected_tab == "Exploración":
    st.markdown("### Distribución general de precios")
    fig = px.histogram(df, x="$Final", nbins=40)
    promedio = df["$Final"].mean()
    fig.add_vline(x=promedio, line_dash="dash", line_color="#E11D48",
                  annotation_text=f"Promedio: ${promedio:,.0f}")
    fig.update_layout(xaxis_title="Precio Final ($/Kg)", yaxis_title="Frecuencia")
    fig = aplicar_estilo_grafico(fig)
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    top5 = df["Sexo"].value_counts().head(5).index
    df_top5 = df[df["Sexo"].isin(top5)].copy()
    df_top5["Etiqueta"] = df_top5["Sexo"].map(etiqueta_sexo)

    with col1:
        st.markdown("### Peso vs. Precio (Top 5 categorías)")
        fig2 = px.scatter(df_top5, x="P.Prom", y="$Final", color="Etiqueta",
                          opacity=0.6, trendline="ols",
                          labels={"P.Prom": "Peso Promedio (Kg)", "$Final": "Precio Final ($/Kg)"})
        fig2 = aplicar_estilo_grafico(fig2)
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.markdown("### Rango de precios por categoría (Top 5)")
        orden = df_top5.groupby("Etiqueta")["$Final"].median().sort_values(ascending=False).index
        fig3 = px.box(df_top5, x="Etiqueta", y="$Final", color="Etiqueta",
                      category_orders={"Etiqueta": list(orden)})
        fig3.update_layout(showlegend=False, xaxis_title="Categoría", yaxis_title="Precio Final ($/Kg)")
        fig3 = aplicar_estilo_grafico(fig3)
        st.plotly_chart(fig3, use_container_width=True)

# ---------------------------------------------------------
# TAB 3: CORRELACIÓN
# ---------------------------------------------------------
elif selected_tab == "Correlación":
    st.markdown("### 📊 Matriz de Correlación Multivariable (Heatmap)")
    
    cols_heatmap = ["$Final", "$Base", "P.Prom", "Cant.", "Margen_Puja", "Hubo_Puja", "Hora_Num"]
    labels_heatmap = ["Precio Final", "Precio Base", "Peso Prom.", "Cantidad", "Margen Puja", "Hubo Puja", "Hora Entrada"]
    
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
    cols_num = ["Cant.", "P.Prom", "$Base", "$Final"] + [c for c in df_dummies.columns if c.startswith("Sexo_")]
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
# TAB 4: PREDICTOR
# ---------------------------------------------------------
elif selected_tab == "Predictor":
    @st.cache_resource
    def entrenar_modelos(df_in: pd.DataFrame):
        d = df_in.copy()
        d["Hora_Entrada"] = "Hora_" + d["Hora_Entrada"].astype(str)

        X_simple = d[["P.Prom"]]
        y = d["$Final"]
        X_tr, X_te, y_tr, y_te = train_test_split(X_simple, y, test_size=0.2, random_state=42)
        modelo_simple = LinearRegression().fit(X_tr, y_tr)
        r2_simple = r2_score(y_te, modelo_simple.predict(X_te))

        d_model = pd.get_dummies(d, columns=["Sexo", "Procedencia", "Hora_Entrada"], drop_first=True)
        cols_sexo = [c for c in d_model.columns if c.startswith("Sexo_")]
        cols_proc = [c for c in d_model.columns if c.startswith("Procedencia_")]
        cols_hora = [c for c in d_model.columns if c.startswith("Hora_Entrada_")]
        columnas_x = ["P.Prom", "Cant."] + cols_sexo + cols_proc + cols_hora

        X_multi = d_model[columnas_x]
        X_tr_m, X_te_m, y_tr_m, y_te_m = train_test_split(X_multi, y, test_size=0.2, random_state=42)
        modelo_multi = LinearRegression().fit(X_tr_m, y_tr_m)
        y_pred_m = modelo_multi.predict(X_te_m)
        r2_multi = r2_score(y_te_m, y_pred_m)

        return {
            "modelo_simple": modelo_simple, "r2_simple": r2_simple,
            "modelo_multi": modelo_multi, "r2_multi": r2_multi,
            "columnas_x": columnas_x,
            "sexos": sorted(d["Sexo"].unique()),
            "procedencias": sorted(d["Procedencia"].unique()),
            "horas": sorted(d["Hora_Entrada"].astype(str).unique()),
        }

    modelos = entrenar_modelos(df)

    c1, c2, c3 = st.columns(3)
    c1.metric("R² Regresión Simple", f"{modelos['r2_simple']*100:.1f}%")
    c2.metric("R² Regresión Múltiple", f"{modelos['r2_multi']*100:.1f}%")
    mejora = (modelos["r2_multi"] - modelos["r2_simple"]) / max(modelos["r2_simple"], 1e-6) * 100
    c3.metric("Mejora vs. Simple", f"+{mejora:.0f}%")

    st.markdown("---")
    st.markdown("### 🧮 Calculator Predictor de Precio por Kilo")

    colf1, colf2, colf3, colf4 = st.columns(4)
    with colf1:
        peso_in = st.number_input("Peso Promedio (Kg)", min_value=20, max_value=600, value=150, step=5)
    with colf2:
        cant_in = st.number_input("Cantidad de animales", min_value=1, max_value=50, value=3, step=1)
    with colf3:
        sexo_in = st.selectbox("Categoría (Sexo)", options=modelos["sexos"], format_func=etiqueta_sexo)
    with colf4:
        hora_in = st.selectbox("Hora de entrada", options=modelos["horas"])

    procedencia_in = st.selectbox("Procedencia", options=modelos["procedencias"])

    def predecir_precio(peso, cantidad, sexo, procedencia, hora):
        datos_nuevos = {c: 0 for c in modelos["columnas_x"]}
        datos_nuevos["P.Prom"] = peso
        datos_nuevos["Cant."] = cantidad
        col_sexo = f"Sexo_{sexo}"
        if col_sexo in datos_nuevos:
            datos_nuevos[col_sexo] = 1
        col_proc = f"Procedencia_{procedencia}"
        if col_proc in datos_nuevos:
            datos_nuevos[col_proc] = 1
        col_hora = f"Hora_Entrada_{hora}"
        if col_hora in datos_nuevos:
            datos_nuevos[col_hora] = 1
        df_pred = pd.DataFrame([datos_nuevos])[modelos["columnas_x"]]
        return modelos["modelo_multi"].predict(df_pred)[0]

    if st.button("💰 Calcular precio estimado", type="primary"):
        precio_est = predecir_precio(peso_in, cant_in, sexo_in, procedencia_in, hora_in)
        valor_total = precio_est * peso_in
        cA, cB = st.columns(2)
        cA.metric("Precio estimado por Kg", f"${precio_est:,.0f}")
        cB.metric("Valor estimado por animal", f"${valor_total:,.0f}")

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
                                 mode="lines+markers", name="Histórico", line=dict(color="#0B2265")))
        fig.add_trace(go.Scatter(x=forecast_full.index, y=forecast_full.values,
                                 mode="lines+markers", name="Pronóstico", line=dict(dash="dash", color="#008037")))
        fig.update_layout(xaxis_title="Semana", yaxis_title="Precio Final Promedio ($/Kg)")
        fig = aplicar_estilo_grafico(fig)
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# TAB 6: PROBABILIDAD DE PUJA
# ---------------------------------------------------------
elif selected_tab == "Prob. Puja":
    st.markdown("### Clasificación de probabilidad de Puja (Árbol de Decisión)")
    cols = ["Cant.", "P.Prom", "Procedencia", "$Base", "Hubo_Puja"]
    d = df[cols].copy().dropna()
    d = pd.get_dummies(d, columns=["Procedencia"], drop_first=True)
    X = d.drop(columns=["Hubo_Puja"])
    y = d["Hubo_Puja"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    modelo_arbol = DecisionTreeClassifier(max_depth=3, random_state=42, class_weight="balanced").fit(X_tr, y_tr)

    fig, ax = plt.subplots(figsize=(14, 6))
    plot_tree(modelo_arbol, feature_names=X.columns.tolist(), class_names=["Sin Puja", "Con Puja"],
              filled=True, rounded=True, fontsize=8, ax=ax)
    st.pyplot(fig)

# ---------------------------------------------------------
# TAB 7: COMPRADORES (CLUSTERING)
# ---------------------------------------------------------
elif selected_tab == "Compradores":
    st.markdown("### Segmentación por comportamiento de compra (K-Means)")
    k_sel = st.slider("Número de perfiles (K)", 2, 6, 3)
    
    cols = ["Cant.", "P.Prom", "$Base", "$Final"]
    d = df[cols].copy().dropna()
    d["Margen_Puja"] = d["$Final"] - d["$Base"]
    X = d[["P.Prom", "$Final", "Margen_Puja"]]
    X_scaled = StandardScaler().fit_transform(X)
    modelo = KMeans(n_clusters=k_sel, random_state=10, n_init=10).fit(X_scaled)
    d["Perfil_ID"] = modelo.labels_

    fig = px.scatter(
        d, x="P.Prom", y="$Final", color=d["Perfil_ID"].astype(str),
        labels={"P.Prom": "Peso Promedio (Kg)", "$Final": "Precio Final ($/Kg)", "color": "Perfil"}
    )
    fig = aplicar_estilo_grafico(fig)
    st.plotly_chart(fig, use_container_width=True)

# =========================================================
# FOOTER INSTITUCIONAL REPLICADO (HTML SINTAXIS PLANA)
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
