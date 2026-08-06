"""
Dashboard de Analítica Predictiva - Subasta Ganadera Suganorte (Zarzal, Valle)
Construido con Streamlit a partir del notebook de analítica predictiva.
"""

import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt

# =========================================================
# CONFIGURACIÓN DE PÁGINA Y ESTILO
# =========================================================
st.set_page_config(
    page_title="Suganorte | Analítica de Subasta Ganadera",
    page_icon="🐂",
    layout="wide",
    initial_sidebar_state="expanded",
)

PRIMARY = "#7A3B24"      # cuero / tierra
ACCENT = "#C97A3D"       # terracota ganadero
GREEN = "#4B6E3A"        # pasto
CREAM = "#F6F1E7"        # papel
DARK = "#2E2A24"

st.markdown(f"""
<style>
    /* Fuerza tema claro sin importar el modo del sistema/navegador */
    html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {{
        background-color: {CREAM} !important;
        color: {DARK} !important;
    }}

    /* Texto general: párrafos, labels, captions, listas, spans */
    p, span, label, li, div, small,
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stCaptionContainer"], [data-testid="stCaption"],
    [data-testid="stWidgetLabel"] p {{
        color: {DARK} !important;
    }}

    h1, h2, h3, h4, h5, h6,
    [data-testid="stMarkdownContainer"] h1,
    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3 {{
        color: {DARK} !important;
        font-family: 'Georgia', serif;
    }}

    /* Sidebar: fondo claro y texto oscuro, forzado */
    section[data-testid="stSidebar"],
    [data-testid="stSidebarContent"] {{
        background-color: #EFE7D8 !important;
        border-right: 1px solid #DDD0B8;
    }}
    section[data-testid="stSidebar"] * {{
        color: {DARK} !important;
    }}

    /* Métricas (KPIs) */
    div[data-testid="stMetric"] {{
        background-color: white !important;
        border: 1px solid #E4D8C0;
        border-left: 5px solid {ACCENT};
        border-radius: 8px;
        padding: 10px 16px;
    }}
    div[data-testid="stMetric"] * {{ color: {DARK} !important; }}
    div[data-testid="stMetricValue"] {{ color: {PRIMARY} !important; font-weight: 700; }}

    /* Inputs: selects, multiselect, date input, number input, sliders */
    [data-baseweb="select"] > div, [data-baseweb="input"] > div,
    input, textarea {{
        background-color: white !important;
        color: {DARK} !important;
    }}
    [data-baseweb="tag"] {{ color: white !important; }}

    /* Pestañas (tabs) */
    .stTabs [data-baseweb="tab-list"] {{ gap: 4px; }}
    .stTabs [data-baseweb="tab"] {{
        background-color: #EFE7D8 !important;
        border-radius: 6px 6px 0 0;
        padding: 8px 16px;
    }}
    .stTabs [data-baseweb="tab"] p {{ color: {DARK} !important; }}
    .stTabs [aria-selected="true"] {{
        background-color: {ACCENT} !important;
    }}
    .stTabs [aria-selected="true"] p {{ color: white !important; }}

    /* Banner superior */
    .banner {{
        background: linear-gradient(90deg, {PRIMARY}, {ACCENT});
        padding: 22px 28px; border-radius: 10px; margin-bottom: 18px;
    }}
    .banner h1, .banner p {{ color: white !important; }}
    .banner h1 {{ margin: 0; font-size: 30px; }}
    .banner p {{ margin: 4px 0 0 0; font-size: 15px; }}

    /* Tablas / dataframes */
    [data-testid="stDataFrame"] {{ background-color: white !important; }}

    /* Alertas informativas (st.info) legibles sobre fondo claro */
    div[data-testid="stAlert"] p {{ color: {DARK} !important; }}
</style>
""", unsafe_allow_html=True)

SEXO_LABELS = {
    "HL": "Hembra de Levante", "ML": "Macho de Levante", "VH": "Vaca Horra",
    "HV": "Hembra de Vientre", "TR": "Ternero(a)", "MC": "Macho de Ceba",
    "VI": "Vaca Industrial", "VP": "Vaca Parida", "TO": "Toro",
    "BF": "Búfala", "BH": "Búfalo", "TI": "Toro/Otro",
}


def etiqueta_sexo(codigo: str) -> str:
    return f"{codigo} · {SEXO_LABELS.get(codigo, codigo)}"


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


df_total = cargar_datos("data.csv")

# =========================================================
# SIDEBAR - FILTROS GLOBALES
# =========================================================
st.sidebar.title("🐂 Suganorte")
st.sidebar.caption("Analítica Predictiva de Subasta Ganadera — Zarzal, Valle")
st.sidebar.markdown("---")
st.sidebar.subheader("Filtros")

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
st.sidebar.caption("Fuente: Registros de subasta Suganorte · ≈9 meses de ventas")

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
# BANNER
# =========================================================
st.markdown("""
<div class="banner">
    <h1>🐂 Analítica Predictiva — Subasta Ganadera Suganorte</h1>
    <p>Precio por Kilo · Comportamiento de Puja · Segmentación de Compradores · Pronóstico de Precios</p>
</div>
""", unsafe_allow_html=True)

if df.empty:
    st.warning("No hay datos para los filtros seleccionados.")
    st.stop()

# =========================================================
# TABS
# =========================================================
tab_resumen, tab_eda, tab_corr, tab_reg, tab_ts, tab_arbol, tab_cluster = st.tabs(
    ["📊 Resumen", "🔍 Exploración", "🔗 Correlación", "📈 Regresión y Predictor",
     "📅 Pronóstico", "🌳 Prob. de Puja", "🎯 Perfiles de Compradores"]
)

# ---------------------------------------------------------
# TAB 1: RESUMEN
# ---------------------------------------------------------
with tab_resumen:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Lotes vendidos", f"{len(df):,}")
    c2.metric("Animales totales", f"{int(df['Cant.'].sum()):,}")
    c3.metric("Precio Prom. ($/Kg)", f"${df['$Final'].mean():,.0f}")
    c4.metric("Peso Prom. (Kg)", f"{df['P.Prom'].mean():,.1f}")
    tasa_puja = df["Hubo_Puja"].mean() * 100
    c5.metric("Lotes con Puja", f"{tasa_puja:,.1f}%")

    st.markdown("### Evolución del precio final por Kg")
    serie_diaria = df.groupby("Fecha_TS")["$Final"].mean().reset_index()
    fig = px.line(serie_diaria, x="Fecha_TS", y="$Final", markers=True,
                  color_discrete_sequence=[ACCENT])
    fig.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                       xaxis_title="Fecha", yaxis_title="Precio Final Promedio ($/Kg)")
    st.plotly_chart(fig, width='stretch')

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### Volumen por Categoría (Sexo)")
        vol_sexo = df["Sexo"].value_counts().reset_index()
        vol_sexo.columns = ["Sexo", "Lotes"]
        vol_sexo["Etiqueta"] = vol_sexo["Sexo"].map(lambda s: etiqueta_sexo(s))
        fig2 = px.bar(vol_sexo, x="Lotes", y="Etiqueta", orientation="h",
                      color_discrete_sequence=[GREEN])
        fig2.update_layout(plot_bgcolor="white", paper_bgcolor="white", yaxis_title="")
        st.plotly_chart(fig2, width='stretch')
    with col_b:
        st.markdown("### Top 10 Procedencias por volumen")
        vol_proc = df["Procedencia"].value_counts().head(10).reset_index()
        vol_proc.columns = ["Procedencia", "Lotes"]
        fig3 = px.bar(vol_proc, x="Lotes", y="Procedencia", orientation="h",
                      color_discrete_sequence=[PRIMARY])
        fig3.update_layout(plot_bgcolor="white", paper_bgcolor="white", yaxis_title="")
        st.plotly_chart(fig3, width='stretch')

# ---------------------------------------------------------
# TAB 2: EDA
# ---------------------------------------------------------
with tab_eda:
    st.markdown("### Distribución general de precios")
    fig = px.histogram(df, x="$Final", nbins=40, color_discrete_sequence=[ACCENT])
    promedio = df["$Final"].mean()
    fig.add_vline(x=promedio, line_dash="dash", line_color="red",
                  annotation_text=f"Promedio: ${promedio:,.0f}")
    fig.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                       xaxis_title="Precio Final ($/Kg)", yaxis_title="Frecuencia")
    st.plotly_chart(fig, width='stretch')

    col1, col2 = st.columns(2)
    top5 = df["Sexo"].value_counts().head(5).index
    df_top5 = df[df["Sexo"].isin(top5)].copy()
    df_top5["Etiqueta"] = df_top5["Sexo"].map(etiqueta_sexo)

    with col1:
        st.markdown("### Peso vs. Precio (Top 5 categorías)")
        fig2 = px.scatter(df_top5, x="P.Prom", y="$Final", color="Etiqueta",
                          opacity=0.6, trendline="ols",
                          labels={"P.Prom": "Peso Promedio (Kg)", "$Final": "Precio Final ($/Kg)"})
        fig2.update_layout(plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig2, width='stretch')

    with col2:
        st.markdown("### Rango de precios por categoría (Top 5)")
        orden = df_top5.groupby("Etiqueta")["$Final"].median().sort_values(ascending=False).index
        fig3 = px.box(df_top5, x="Etiqueta", y="$Final", color="Etiqueta",
                      category_orders={"Etiqueta": list(orden)})
        fig3.update_layout(plot_bgcolor="white", paper_bgcolor="white", showlegend=False,
                           xaxis_title="Categoría", yaxis_title="Precio Final ($/Kg)")
        st.plotly_chart(fig3, width='stretch')

    st.info(
        "💡 **Interpretación:** entre más pese el lote, menor suele ser el valor por Kg (aunque el valor final "
        "del lote sea alto). Machos y Hembras de Levante (ML/HL) tienden a alcanzar el mayor precio por Kg, "
        "mientras que Vacas Horras (VH) y Vacas Industriales (VI) tienden al precio más bajo."
    )

# ---------------------------------------------------------
# TAB 3: CORRELACIÓN
# ---------------------------------------------------------
with tab_corr:
    st.markdown("### Matriz de correlación contra el Precio Final")
    df_dummies = pd.get_dummies(df, columns=["Sexo"], drop_first=True)
    cols_num = ["Cant.", "P.Prom", "$Base", "$Final"] + [c for c in df_dummies.columns if c.startswith("Sexo_")]
    corr_obj = df_dummies[cols_num].corr(numeric_only=True)[["$Final"]].sort_values(by="$Final", ascending=False)
    corr_obj = corr_obj.drop(index="$Final")

    fig = px.bar(corr_obj, x="$Final", y=corr_obj.index, orientation="h",
                color="$Final", color_continuous_scale=["#C0392B", "#EEE", "#27AE60"],
                labels={"$Final": "Correlación con Precio Final", "y": "Variable"})
    fig.update_layout(plot_bgcolor="white", paper_bgcolor="white", height=500)
    st.plotly_chart(fig, width='stretch')

    st.markdown("### Mapa de calor — variables numéricas")
    fig_hm = px.imshow(df_dummies[cols_num].corr(numeric_only=True).round(2),
                       text_auto=True, color_continuous_scale="RdYlGn", aspect="auto")
    fig_hm.update_layout(paper_bgcolor="white")
    st.plotly_chart(fig_hm, width='stretch')

# ---------------------------------------------------------
# TAB 4: REGRESIÓN + PREDICTOR
# ---------------------------------------------------------
@st.cache_resource
def entrenar_modelos(df_in: pd.DataFrame):
    d = df_in.copy()
    d["Hora_Entrada"] = "Hora_" + d["Hora_Entrada"].astype(str)

    # --- Simple ---
    X_simple = d[["P.Prom"]]
    y = d["$Final"]
    X_tr, X_te, y_tr, y_te = train_test_split(X_simple, y, test_size=0.2, random_state=42)
    modelo_simple = LinearRegression().fit(X_tr, y_tr)
    r2_simple = r2_score(y_te, modelo_simple.predict(X_te))

    # --- Múltiple ---
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


with tab_reg:
    modelos = entrenar_modelos(df)

    c1, c2, c3 = st.columns(3)
    c1.metric("R² Regresión Simple", f"{modelos['r2_simple']*100:.1f}%")
    c2.metric("R² Regresión Múltiple", f"{modelos['r2_multi']*100:.1f}%")
    mejora = (modelos["r2_multi"] - modelos["r2_simple"]) / max(modelos["r2_simple"], 1e-6) * 100
    c3.metric("Mejora vs. Simple", f"+{mejora:.0f}%")

    st.caption(
        f"MAE: ± ${modelos['mae_multi']:,.0f} por Kg · RMSE: ± ${modelos['rmse_multi']:,.0f} por Kg "
        "(regresión múltiple, sobre datos de prueba)."
    )

    st.markdown("### Impacto económico por variable (Top 15)")
    top15 = modelos["df_impactos"].head(15).sort_values("Impacto_Pesos")
    colores = ["#C0392B" if v < 0 else "#27AE60" for v in top15["Impacto_Pesos"]]
    fig = go.Figure(go.Bar(
        x=top15["Impacto_Pesos"], y=top15["Variable"], orientation="h",
        marker_color=colores,
        text=[f"{'+' if v > 0 else ''}${v:,.0f}" for v in top15["Impacto_Pesos"]],
        textposition="outside",
    ))
    fig.add_vline(x=0, line_color="black", line_dash="dash")
    fig.update_layout(plot_bgcolor="white", paper_bgcolor="white", height=500,
                      xaxis_title="Impacto en $ COP", yaxis_title="")
    st.plotly_chart(fig, width='stretch')

    st.markdown("---")
    st.markdown("## 🧮 Predictor de Precio por Kilo")
    st.caption("Simula un lote y estima su precio por Kg según el modelo de regresión múltiple.")

    colf1, colf2, colf3, colf4 = st.columns(4)
    with colf1:
        peso_in = st.number_input("Peso Promedio (Kg)", min_value=20, max_value=600, value=150, step=5)
    with colf2:
        cant_in = st.number_input("Cantidad de animales en el lote", min_value=1, max_value=50, value=3, step=1)
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
        st.caption(
            f"Estimación con margen de error aproximado de ± ${modelos['mae_multi']:,.0f} por Kg. "
            "Basado en 9 meses de histórico de subasta; no incorpora raza, biotipo ni preñez."
        )

# ---------------------------------------------------------
# TAB 5: SERIES DE TIEMPO
# ---------------------------------------------------------
with tab_ts:
    st.markdown("### Pronóstico de precio semanal (Holt-Winters)")

    df_ts = df_total.copy().set_index("Fecha_TS").sort_index()
    precio_semanal = df_ts["$Final"].resample("W").mean().ffill()

    if len(precio_semanal) < 8:
        st.warning("No hay suficientes semanas de datos para un pronóstico confiable con los filtros actuales.")
    else:
        semanas_prueba = min(4, max(1, len(precio_semanal) // 5))
        train_hw = precio_semanal[:-semanas_prueba]
        test_hw = precio_semanal[-semanas_prueba:]

        modelo_eval = ExponentialSmoothing(
            train_hw, trend="add", seasonal=None, initialization_method="estimated"
        ).fit()
        pred_eval = modelo_eval.forecast(semanas_prueba)

        mae_hw = mean_absolute_error(test_hw, pred_eval)
        rmse_hw = np.sqrt(mean_squared_error(test_hw, pred_eval))
        mape_hw = np.mean(np.abs((test_hw - pred_eval) / test_hw)) * 100

        c1, c2, c3 = st.columns(3)
        c1.metric("MAE", f"± ${mae_hw:,.0f}/Kg")
        c2.metric("RMSE", f"± ${rmse_hw:,.0f}/Kg")
        c3.metric("MAPE", f"± {mape_hw:.1f}%")

        semanas_futuro = st.slider("Semanas a pronosticar hacia adelante", 2, 12, 4)
        modelo_full = ExponentialSmoothing(
            precio_semanal, trend="add", seasonal=None, initialization_method="estimated"
        ).fit()
        forecast_full = modelo_full.forecast(semanas_futuro)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=precio_semanal.index, y=precio_semanal.values,
                                 mode="lines+markers", name="Histórico", line=dict(color=PRIMARY)))
        fig.add_trace(go.Scatter(x=forecast_full.index, y=forecast_full.values,
                                 mode="lines+markers", name="Pronóstico", line=dict(color=ACCENT, dash="dash")))
        fig.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                          xaxis_title="Semana", yaxis_title="Precio Final Promedio ($/Kg)")
        st.plotly_chart(fig, width='stretch')

        st.info(
            "💡 Se eligió **Holt-Winters** (tendencia aditiva, sin componente estacional) sobre SARIMA porque "
            "el histórico disponible (~9 meses) es corto para exigir estacionalidad anual y la serie no es "
            "estrictamente estacionaria por la volatilidad propia del mercado ganadero."
        )

# ---------------------------------------------------------
# TAB 6: ÁRBOL DE DECISIÓN (PROB. DE PUJA)
# ---------------------------------------------------------
@st.cache_resource
def entrenar_arbol(df_in: pd.DataFrame):
    cols = ["Cant.", "P.Prom", "Procedencia", "$Base", "Hubo_Puja"]
    d = df_in[cols].copy().dropna()
    d = pd.get_dummies(d, columns=["Procedencia"], drop_first=True)
    X = d.drop(columns=["Hubo_Puja"])
    y = d["Hubo_Puja"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    modelo = DecisionTreeClassifier(max_depth=3, random_state=42, class_weight="balanced").fit(X_tr, y_tr)
    acc = modelo.score(X_te, y_te)
    return modelo, X.columns.tolist(), acc


with tab_arbol:
    st.markdown("### ¿Qué tan probable es que un lote reciba puja (precio final > precio base)?")
    modelo_arbol, cols_arbol, acc_arbol = entrenar_arbol(df)
    st.metric("Precisión del árbol (test)", f"{acc_arbol*100:.1f}%")

    fig, ax = plt.subplots(figsize=(16, 7))
    plot_tree(modelo_arbol, feature_names=cols_arbol, class_names=["Sin Puja", "Con Puja"],
              filled=True, rounded=True, fontsize=8, ax=ax)
    st.pyplot(fig)

    st.info(
        "💡 **Lectura del árbol:** el peso del lote (~181.5 Kg) suele marcar el punto de inflexión entre "
        "animales de levante y de destete. Lotes con precio base moderado tienden a generar más puja; "
        "cuando el precio base ya arranca alto, el comprador pierde interés y la probabilidad de puja cae."
    )

# ---------------------------------------------------------
# TAB 7: CLUSTERING (PERFILES DE COMPRADORES)
# ---------------------------------------------------------
@st.cache_resource
def entrenar_clusters(df_in: pd.DataFrame, k: int):
    cols = ["Cant.", "P.Prom", "$Base", "$Final"]
    d = df_in[cols].copy().dropna()
    d["Margen_Puja"] = d["$Final"] - d["$Base"]
    X = d[["P.Prom", "$Final", "Margen_Puja"]]
    X_scaled = StandardScaler().fit_transform(X)
    modelo = KMeans(n_clusters=k, random_state=10, n_init=10).fit(X_scaled)
    d["Perfil_ID"] = modelo.labels_
    d["Margen_Puja_Abs"] = d["Margen_Puja"].abs()
    resumen = d.groupby("Perfil_ID")[["P.Prom", "$Final", "Margen_Puja", "Cant."]].mean().round(1)
    return d, resumen


with tab_cluster:
    st.markdown("### Segmentación de lotes por comportamiento de compra (K-Means)")
    k_sel = st.slider("Número de perfiles (K)", 2, 6, 3)
    d_kmeans, resumen = entrenar_clusters(df, k_sel)

    fig = px.scatter(
        d_kmeans, x="P.Prom", y="$Final", color=d_kmeans["Perfil_ID"].astype(str),
        size="Margen_Puja_Abs", size_max=30, opacity=0.65,
        hover_data={"Margen_Puja": True},
        labels={"P.Prom": "Peso Promedio (Kg)", "$Final": "Precio Final ($/Kg)", "color": "Perfil"},
    )
    fig.update_layout(plot_bgcolor="white", paper_bgcolor="white", height=550)
    st.plotly_chart(fig, width='stretch')

    st.markdown("### Resumen por perfil")
    st.dataframe(resumen.rename(columns={
        "P.Prom": "Peso Prom. (Kg)", "$Final": "Precio Prom. ($/Kg)",
        "Margen_Puja": "Margen de Puja ($)", "Cant.": "Animales por Lote"
    }), width='stretch')

    if k_sel == 3:
        st.info(
            "💡 **Perfiles de referencia (K=3):** *Destete Precoz Premium* (lotes pequeños, peso bajo, precio "
            "alto, mucha puja) · *Mercado de Frigorífico* (peso alto, margen de puja bajo — compradores con "
            "calculadora) · *Destete Numeroso y Homogéneo* (lotes medianos y parejos, puja moderada)."
        )

# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------
st.markdown("---")
st.caption(
    "Proyecto de Analítica Predictiva — Suganorte (Zarzal, Valle) · "
    "Integrantes: Jeferson Balcazar Gomez, Carlos Arturo Agudelo Garcia, Milton Vanegas Delgado."
)
