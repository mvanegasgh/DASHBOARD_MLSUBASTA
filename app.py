"""
Dashboard de Analítica Predictiva - Subasta Ganadera Suganorte S.A. (Zarzal, Valle)
Interfaz Institucional Oficial alineada a la marca Suganorte S.A.
"""

import re
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
# RECURSOS DE MARCA (LOGOS EN VECTOR SVG SEGURO)
# =========================================================
# Logo Color para Sidebar
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

# Logo Blanco para Banner
LOGO_WHITE_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 450 110" height="65">
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
# ESTILOS CSS CORREGIDOS (CORRECCIÓN DE FILTROS Y TEMA)
# =========================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    /* BASE DE LA APLICACIÓN */
    html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
        background-color: #F8FAFC !important;
        color: #0F172A !important;
        font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
    }

    h1, h2, h3, h4, h5, h6, p, span, label, input, button {
        font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
    }

    /* BANNER INSTITUCIONAL ENCABEZADO */
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

    /* BARRA LATERAL (SIDEBAR) */
    section[data-testid="stSidebar"], [data-testid="stSidebarContent"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E2E8F0 !important;
    }
    
    /* CORRECCIÓN DE ESTILOS EN INPUTS Y FILTROS */
    div[data-baseweb="select"] > div, 
    div[data-baseweb="input"] > div,
    input {
        background-color: #F1F5F9 !important;
        border-color: #CBD5E1 !important;
        color: #0F172A !important;
        border-radius: 8px !important;
    }
    
    /* REMOVER TONOS CREMAS EN MULTISELECT CHIPS */
    span[data-baseweb="tag"] {
        background-color: #0B2265 !important;
        border-radius: 6px !important;
        border: none !important;
    }
    span[data-baseweb="tag"] span {
        color: #FFFFFF !important;
        font-weight: 600 !important;
    }

    /* METRICAS / KPIS */
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
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        background-color: #006028 !important;
    }

    .stMain h3 {
        color: #0B2265 !important;
        font-weight: 700 !important;
        font-size: 1.2rem !important;
        margin-top: 1.5rem !important;
        margin-bottom: 1rem !important;
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
# CARGA Y LIMPIEZA DE DATOS
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
except Exception as e:
    st.error("Asegúrate de tener el archivo 'data.csv' en el directorio de la aplicación.")
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
# SIDEBAR - LOGO Y FILTROS
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

# Aplicar filtros
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
# ENCABEZADO PRINCIPAL
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
# MENÚ DE NAVEGACIÓN
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

es_tab_resumen = selected_tab == "Resumen"
es_tab_eda = selected_tab == "Exploración"
es_tab_corr = selected_tab == "Correlación"
es_tab_reg = selected_tab == "Predictor"
es_tab_ts = selected_tab == "Pronóstico"
es_tab_arbol = selected_tab == "Prob. Puja"
es_tab_cluster = selected_tab == "Compradores"

# ---------------------------------------------------------
# TAB 1: RESUMEN
# ---------------------------------------------------------
if es_tab_resumen:
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
        vol_sexo["Etiqueta"] = vol_sexo["Sexo"].map(lambda s: etiqueta_sexo(s))
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
if es_tab_eda:
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
if es_tab_corr:
    st.markdown("### Matriz de correlación contra el Precio Final")
    df_dummies = pd.get_dummies(df, columns=["Sexo"], drop_first=True)
    cols_num = ["Cant.", "P.Prom", "$Base", "$Final"] + [c for c in df_dummies.columns if c.startswith("Sexo_")]
    corr_obj = df_dummies[cols_num].corr(numeric_only=True)[["$Final"]].sort_values(by="$Final", ascending=False)
    corr_obj = corr_obj.drop(index="$Final")

    fig = px.bar(corr_obj, x="$Final", y=corr_obj.index, orientation="h",
                color="$Final", color_continuous_scale=["#E11D48", "#F8FAFC", "#008037"],
                labels={"$Final": "Correlación con Precio Final", "y": "Variable"})
    fig.update_layout(height=500)
    fig = aplicar_estilo_grafico(fig)
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# TAB 4: REGRESIÓN + PREDICTOR
# ---------------------------------------------------------
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
    mae_multi = mean_absolute_error(y_te_m, y_pred_m)
    rmse_multi = np.sqrt(mean_squared_error(y_te_m, y_pred_m))

    df_impactos = pd.DataFrame({"Variable": columnas_x, "Impacto_Pesos": modelo_multi.coef_})
    df_impactos["Magnitud"] = df_impactos["Impacto_Pesos"].abs()
    df_impactos = df_impactos.sort_values(by="Magnitud", ascending=False)

    return {
        "modelo_simple": modelo_simple, "r2_simple": r2_simple,
        "modelo_multi": modelo_multi, "r2_multi": r2_multi,
        "mae_multi": mae_multi, "rmse_multi": rmse_multi,
        "columnas_x": columnas_x, "df_impactos": df_impactos,
        "sexos": sorted(d["Sexo"].unique()),
        "procedencias": sorted(d["Procedencia"].unique()),
        "horas": sorted(d["Hora_Entrada"].astype(str).unique()),
    }

if es_tab_reg:
    modelos = entrenar_modelos(df)

    c1, c2, c3 = st.columns(3)
    c1.metric("R² Regresión Simple", f"{modelos['r2_simple']*100:.1f}%")
    c2.metric("R² Regresión Múltiple", f"{modelos['r2_multi']*100:.1f}%")
    mejora = (modelos["r2_multi"] - modelos["r2_simple"]) / max(modelos["r2_simple"], 1e-6) * 100
    c3.metric("Mejora vs. Simple", f"+{mejora:.0f}%")

    st.markdown("---")
    st.markdown("## 🧮 Predictor de Precio por Kilo")

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
        col_hora = f"Hora_Entrada_Hora_{hora}"
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
# TAB 5: SERIES DE TIEMPO
# ---------------------------------------------------------
if es_tab_ts:
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
# TAB 6: ÁRBOL DE DECISIÓN
# ---------------------------------------------------------
if es_tab_arbol:
    st.markdown("### Clasificación de probabilidad de Puja")
    cols = ["Cant.", "P.Prom", "Procedencia", "$Base", "Hubo_Puja"]
    d = df[cols].copy().dropna()
    d = pd.get_dummies(d, columns=["Procedencia"], drop_first=True)
    X = d.drop(columns=["Hubo_Puja"])
    y = d["Hubo_Puja"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    modelo_arbol = DecisionTreeClassifier(max_depth=3, random_state=42, class_weight="balanced").fit(X_tr, y_tr)

    fig, ax = plt.subplots(figsize=(16, 7))
    plot_tree(modelo_arbol, feature_names=X.columns.tolist(), class_names=["Sin Puja", "Con Puja"],
              filled=True, rounded=True, fontsize=8, ax=ax)
    st.pyplot(fig)

# ---------------------------------------------------------
# TAB 7: CLUSTERING (COMPRADORES)
# ---------------------------------------------------------
if es_tab_cluster:
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

# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------
st.markdown("---")
st.caption(
    "Proyecto de Analítica Predictiva — Suganorte S.A. (Zarzal, Valle) · "
    "Integrantes: Jeferson Balcazar Gomez, Carlos Arturo Agudelo Garcia, Milton Vanegas Delgado."
)
