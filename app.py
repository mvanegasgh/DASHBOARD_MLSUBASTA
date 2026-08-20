import threading
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
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from flask import Flask, request, jsonify
from rapidfuzz import process, fuzz

# =========================================================
# 1. DICCIONARIO MAESTRO & LÓGICA DE FUZZY MATCHING
# =========================================================
DICCIONARIO_PROCEDENCIAS = {
    # Valle del Cauca
    "ZARZAL": ["ZARZAL", "SARTAL", "ZARSAL", "ZARZAL VALLE", "SARZAL"],
    "BUGALAGRANDE": ["BUGALAGRANDE", "BUGA LA GRANDE", "BUGALAGRAND", "BUGA GRANDE"],
    "SEVILLA": ["SEVILLA", "SEVILA", "SEVILLA VALLE"],
    "ROLDANILLO": ["ROLDANILLO", "ROLDANILO", "ROLDANILLO VALLE"],
    "CARTAGO": ["CARTAGO", "CARTAGO VALLE"],
    "TULUA": ["TULUA", "TULUÁ", "TULUA VALLE"],
    "CAICEDONIA": ["CAICEDONIA", "CAICEDONIA VALLE"],
    "LA VICTORIA": ["LA VICTORIA", "VICTORIA VALLE", "VICTORIA"],
    "OBANDO": ["OBANDO", "OBANDO VALLE"],
    "BOLIVAR": ["BOLIVAR", "BOLIVAR VALLE"],
    "VERSALLES": ["VERSALLES", "VERSALLES VALLE"],
    "TORO": ["TORO", "TORO VALLE"],

    # Quindío
    "ARMENIA": ["ARMENIA", "ARMENIA QUINDIO"],
    "TEBAIDA": ["LA TEBAIDA", "TEBAIDA", "TEBAIDA QUINDIO"],
    "MONTENEGRO": ["MONTENEGRO", "MONTENEGRO QUINDIO"],
    "QUIMBAYA": ["QUIMBAYA", "QUIMBAYA QUINDIO"],
    "CALARCA": ["CALARCA", "CALARCÁ"],

    # Risaralda
    "PEREIRA": ["PEREIRA", "PEREIRA RISARALDA"],
    "DOSQUEBRADAS": ["DOSQUEBRADAS", "DOS QUEBRADAS"],
    "VIRGINIA": ["LA VIRGINIA", "VIRGINIA"],
    "BELEN DE UMBRIA": ["BELEN DE UMBRIA", "BELEN"]
}

MAPEO_INVERSO = {}
for canonical, variaciones in DICCIONARIO_PROCEDENCIAS.items():
    for var in variaciones:
        MAPEO_INVERSO[var.upper()] = canonical

def normalizar_texto(texto: str) -> str:
    if not texto or pd.isna(texto):
        return ""
    texto = str(texto).upper().strip()
    texto = re.sub(r'[^\w\s]', '', texto)
    return re.sub(r'\s+', ' ', texto)

def estandarizar_procedencia(entrada: str, umbral_similitud: int = 75) -> dict:
    limpio = normalizar_texto(entrada)
    if not limpio:
        return {"original": entrada, "estandarizado": "DESCONOCIDA", "match_score": 0, "metodo": "Vacio"}

    if limpio in MAPEO_INVERSO:
        return {
            "original": entrada,
            "estandarizado": MAPEO_INVERSO[limpio],
            "match_score": 100,
            "metodo": "Exacto"
        }

    todas_variaciones = list(MAPEO_INVERSO.keys())
    resultado = process.extractOne(limpio, todas_variaciones, scorer=fuzz.WRatio)

    if resultado and resultado[1] >= umbral_similitud:
        mejor_coincidencia = resultado[0]
        return {
            "original": entrada,
            "estandarizado": MAPEO_INVERSO[mejor_coincidencia],
            "match_score": float(resultado[1]),
            "metodo": "Fuzzy"
        }

    return {
        "original": entrada,
        "estandarizado": limpio,
        "match_score": float(resultado[1]) if resultado else 0.0,
        "metodo": "Original Sin Coincidencia"
    }

def construir_terminos_busqueda(procedencia_filtro: str, df_ref: pd.DataFrame) -> tuple:
    estandar_res = estandarizar_procedencia(procedencia_filtro)
    canonical = estandar_res["estandarizado"]
    procedencias_df = df_ref["Procedencia_Original"].dropna().unique() if "Procedencia_Original" in df_ref.columns else df_ref["Procedencia"].dropna().unique()

    coincidencias = [
        p for p in procedencias_df
        if estandarizar_procedencia(p)["estandarizado"] == canonical
    ]
    return coincidencias, canonical

# =========================================================
# 2. ENDPOINTS EN FLASK DE CONSULTA Y ESTANDARIZACIÓN
# =========================================================
flask_app = Flask("SuganorteAPI")

@flask_app.route("/api/v1/estandarizar", methods=["POST"])
def endpoint_estandarizar():
    data = request.get_json(silent=True) or {}
    if "procedencia" not in data:
        return jsonify({"error": "Parámetro 'procedencia' es requerido"}), 400
    return jsonify(estandarizar_procedencia(data["procedencia"])), 200

@flask_app.route("/api/v1/buscar", methods=["GET"])
def endpoint_buscar():
    query = request.args.get("q", "").strip()
    if not query:
        return jsonify({"error": "Parámetro 'q' es requerido"}), 400

    try:
        df_temp = pd.read_csv("data.csv")
    except Exception as e:
        return jsonify({"error": f"No se pudo cargar data.csv: {str(e)}"}), 500

    terminos, canonical = construir_terminos_busqueda(query, df_temp)
    df_filtrado = df_temp[df_temp["Procedencia"].isin(terminos)]

    return jsonify({
        "busqueda_original": query,
        "categoria_estandar": canonical,
        "variantes_encontradas": terminos,
        "total_registros": len(df_filtrado),
        "resultados": df_filtrado.head(50).to_dict(orient="records")
    }), 200

def ejecutar_flask():
    try:
        flask_app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
    except Exception:
        pass

if "flask_thread" not in st.session_state:
    thread = threading.Thread(target=ejecutar_flask, daemon=True)
    thread.start()
    st.session_state["flask_thread"] = True

# =========================================================
# 3. CONFIGURACIÓN STREAMLIT Y ESTILOS
# =========================================================
st.set_page_config(
    page_title="Suganorte S.A. | Analítica Predictiva & Estandarización",
    page_icon="🐂",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@700;800;900&family=Inter:wght@400;500;600;700&display=swap');
    @import url('https://fonts.googleapis.com/icon?family=Material+Icons|Material+Icons+Outlined|Material+Symbols+Outlined');

    [data-testid="stSidebarCollapseButton"] *,
    i.material-icons, .material-icons, .material-symbols-outlined {
        font-family: 'Material Symbols Outlined', 'Material Icons', sans-serif !important;
    }

    html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"], [data-testid="stMain"] {
        background-color: #F8FAFC !important;
        color: #0F172A !important;
        font-family: 'Inter', sans-serif !important;
    }

    .sidebar-brand { display: flex; align-items: center; gap: 8px; padding: 8px 0; }
    .brand-suganorte { font-family: 'Montserrat', sans-serif; font-weight: 900; font-size: 1.65rem; color: #003399; }
    .brand-sa { font-family: 'Montserrat', sans-serif; font-weight: 800; font-size: 1.1rem; color: #003399; position: relative; }
    .brand-sa::after { content: "🔨"; font-size: 0.85rem; position: absolute; top: -6px; right: -14px; transform: rotate(20deg); }
    .brand-stripe { height: 4px; width: 100%; max-width: 180px; background: linear-gradient(90deg, #FFD100 0% 40%, #008037 40% 75%, #E11D48 75% 100%); border-radius: 2px; margin: 4px 0 12px 0; }

    .suganorte-header-container { background: #003399; border-radius: 12px; overflow: hidden; margin-bottom: 24px; box-shadow: 0 4px 15px rgba(0, 51, 153, 0.2); }
    .tricolor-stripe { height: 6px; background: linear-gradient(90deg, #FFD100 0% 33%, #008037 33% 66%, #E11D48 66% 100%); width: 100%; }
    .suganorte-banner-body { padding: 24px 32px; display: flex; align-items: center; justify-content: space-between; gap: 20px; }
    .suganorte-title { color: #FFFFFF !important; margin: 0 !important; font-size: 1.75rem !important; font-weight: 800 !important; }
    .suganorte-subtitle { color: #FFD100 !important; margin-top: 6px !important; margin-bottom: 0 !important; font-size: 0.95rem !important; font-weight: 600 !important; }

    div[data-testid="stMetric"] { background-color: #FFFFFF !important; border: 1px solid #E2E8F0 !important; border-top: 4px solid #008037 !important; border-radius: 10px !important; padding: 16px 20px !important; }
    div[data-testid="stMetricValue"] { color: #003399 !important; font-size: 1.8rem !important; font-weight: 800 !important; }

    .stButton>button { background-color: #008037 !important; color: #FFFFFF !important; border-radius: 8px !important; border: none !important; font-weight: 700 !important; padding: 10px 24px !important; }
    .stButton>button:hover { background-color: #006028 !important; }

    .suganorte-footer-container { background-color: #003399; border-top: 5px solid #008037; color: #FFFFFF; padding: 40px 30px 20px 30px; border-radius: 12px; margin-top: 50px; }
    .suganorte-footer-grid { display: grid; grid-template-columns: 1.2fr 1fr 1.2fr 1.5fr; gap: 30px; }
    .footer-title { font-size: 1.25rem; font-weight: 700; margin-bottom: 18px; color: #FFFFFF; }
    .footer-links { list-style: none; padding: 0; margin: 0; }
    .footer-links li { margin-bottom: 10px; font-size: 0.9rem; color: #F1F5F9; }
</style>
""", unsafe_allow_html=True)

theme_colors = ["#003399", "#008037", "#D97706", "#2563EB", "#059669"]

def aplicar_estilo_grafico(fig):
    fig.update_layout(
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
        font=dict(family="Inter, sans-serif", color="#334155"),
        colorway=theme_colors, margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

# =========================================================
# 4. CARGA DE DATOS Y ESTANDARIZACIÓN
# =========================================================
@st.cache_data
def cargar_datos(path="data.csv"):
    df = pd.read_csv(path)
    df["$Base"] = pd.to_numeric(df["$Base"], errors="coerce")
    df["$Final"] = pd.to_numeric(df["$Final"], errors="coerce")
    df["Procedencia_Original"] = df["Procedencia"].fillna("Desconocida").astype(str).str.strip()
    
    # Aplicación de Fuzzy Matching al cargar dataset
    res_estandar = df["Procedencia_Original"].apply(lambda x: estandarizar_procedencia(x)["estandarizado"])
    df["Procedencia"] = res_estandar

    df["Fecha_TS"] = pd.to_datetime(df["Fecha"], dayfirst=True, errors="coerce")
    df["Hora_Num"] = pd.to_numeric(df["Entrada"].astype(str).str.split(":").str[0], errors="coerce")
    df["Hora_Entrada"] = df["Entrada"].astype(str).str.split(":").str[0].str.zfill(2)
    df["Margen_Puja"] = df["$Final"] - df["$Base"]
    df["Hubo_Puja"] = (df["$Final"] > df["$Base"]).astype(int)
    return df.dropna(subset=["Fecha_TS"])

try:
    df_total = cargar_datos("data.csv")
except Exception as e:
    st.error(f"Error al cargar 'data.csv': {str(e)}")
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
# 5. BARRA LATERAL (FILTROS)
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
rango_fechas = st.sidebar.date_input("Rango de fechas", value=(fecha_min, fecha_max), min_value=fecha_min, max_value=fecha_max)

sexos_disp = sorted(df_total["Sexo"].unique())
sexos_sel = st.sidebar.multiselect("Categoría (Sexo)", options=sexos_disp, default=sexos_disp, format_func=etiqueta_sexo)

procedencias_disp = sorted(df_total["Procedencia"].unique())
procedencias_sel = st.sidebar.multiselect("Procedencia (Estandarizada)", options=procedencias_disp, default=[])

if isinstance(rango_fechas, tuple) and len(rango_fechas) == 2:
    f_ini, f_fin = rango_fechas
else:
    f_ini, f_fin = fecha_min, fecha_max

mask = (
    (df_total["Fecha_TS"].dt.date >= f_ini) &
    (df_total["Fecha_TS"].dt.date <= f_fin) &
    (df_total["Sexo"].isin(sexos_sel))
)
if procedencias_sel:
    mask &= df_total["Procedencia"].isin(procedencias_sel)

df = df_total[mask].copy()

# =========================================================
# 6. BANNER Y MENU
# =========================================================
st.markdown("""
<div class="suganorte-header-container">
    <div class="tricolor-stripe"></div>
    <div class="suganorte-banner-body">
        <div>
            <h1 class="suganorte-title">Plataforma de Analítica Predictiva</h1>
            <p class="suganorte-subtitle">Subasta Ganadera Suganorte S.A. · Histórico & Pronósticos</p>
        </div>
        <div>
            <span style="color:#FFF; font-weight:900; font-size:1.8rem; font-family:'Montserrat';">Suganorte S.A.</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

if df.empty:
    st.warning("No hay datos para los filtros seleccionados.")
    st.stop()

selected_tab = option_menu(
    menu_title=None,
    options=["Resumen", "Exploración", "Correlación", "Predictor", "Pronóstico", "Prob. Puja", "Compradores", "Motor Fuzzy"],
    icons=["bar-chart-fill", "search", "link-45deg", "calculator-fill", "graph-up-arrow", "diagram-3-fill", "people-fill", "cpu-fill"],
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {"padding": "6px!important", "background-color": "#FFFFFF", "border-radius": "10px", "border": "1px solid #E2E8F0"},
        "icon": {"color": "#003399", "font-size": "14px"},
        "nav-link": {"font-size": "12px", "text-align": "center", "margin": "2px", "color": "#475569", "font-weight": "600", "border-radius": "6px"},
        "nav-link-selected": {"background-color": "#003399", "color": "#FFFFFF", "font-weight": "700"},
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
    c5.metric("Lotes con Puja", f"{df['Hubo_Puja'].mean() * 100:,.1f}%")

    st.markdown("### Evolución del precio final por Kg")
    serie_diaria = df.groupby("Fecha_TS")["$Final"].mean().reset_index()
    fig = px.line(serie_diaria, x="Fecha_TS", y="$Final", markers=True)
    st.plotly_chart(aplicar_estilo_grafico(fig), use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### Volumen por Categoría (Sexo)")
        vol_sexo = df["Sexo"].value_counts().reset_index()
        vol_sexo.columns = ["Sexo", "Lotes"]
        vol_sexo["Etiqueta"] = vol_sexo["Sexo"].map(etiqueta_sexo)
        fig2 = px.bar(vol_sexo, x="Lotes", y="Etiqueta", orientation="h")
        st.plotly_chart(aplicar_estilo_grafico(fig2), use_container_width=True)
    with col_b:
        st.markdown("### Top 10 Procedencias Estandarizadas")
        vol_proc = df["Procedencia"].value_counts().head(10).reset_index()
        vol_proc.columns = ["Procedencia", "Lotes"]
        fig3 = px.bar(vol_proc, x="Lotes", y="Procedencia", orientation="h")
        st.plotly_chart(aplicar_estilo_grafico(fig3), use_container_width=True)

# ---------------------------------------------------------
# TAB 2: EXPLORACIÓN
# ---------------------------------------------------------
elif selected_tab == "Exploración":
    top5 = df["Sexo"].value_counts().head(5).index
    df_top5 = df[df["Sexo"].isin(top5)].copy()
    df_top5["Etiqueta"] = df_top5["Sexo"].map(etiqueta_sexo)

    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.markdown("#### ⚖️ Peso vs. Precio Final por Categoría")
        fig_scatter = px.scatter(df_top5, x="P.Prom", y="$Final", color="Etiqueta", opacity=0.6, trendline="ols")
        st.plotly_chart(aplicar_estilo_grafico(fig_scatter), use_container_width=True)
    with col_g2:
        st.markdown("#### 📦 Precios por Categoría (Boxplot)")
        fig_box = px.box(df_top5, x="Etiqueta", y="$Final", color="Etiqueta")
        st.plotly_chart(aplicar_estilo_grafico(fig_box), use_container_width=True)

    col_g3, col_g4 = st.columns(2)
    with col_g3:
        st.markdown("#### ⏰ Precio Promedio por Hora")
        precio_hora = df.groupby("Hora_Entrada")["$Final"].mean().reset_index()
        fig_hora = px.bar(precio_hora, x="Hora_Entrada", y="$Final", text_auto=".0f")
        st.plotly_chart(aplicar_estilo_grafico(fig_hora), use_container_width=True)
    with col_g4:
        st.markdown("#### 🗺️ Treemap de Procedencias")
        proc_tree = df.groupby("Procedencia").agg(Lotes=("Cant.", "count"), Precio_Prom=("$Final", "mean")).reset_index()
        fig_tree = px.treemap(proc_tree, path=["Procedencia"], values="Lotes", color="Precio_Prom", color_continuous_scale="Greens")
        st.plotly_chart(aplicar_estilo_grafico(fig_tree), use_container_width=True)

# ---------------------------------------------------------
# TAB 3: CORRELACIÓN
# ---------------------------------------------------------
elif selected_tab == "Correlación":
    st.markdown("### 📊 Matriz de Correlación Multivariable")
    cols_heatmap = ["$Final", "$Base", "P.Prom", "Cant.", "Margen_Puja", "Hubo_Puja", "Hora_Num"]
    fig_heatmap = px.imshow(df[cols_heatmap].corr(numeric_only=True), text_auto=".2f", color_continuous_scale=["#E11D48", "#F8FAFC", "#008037"])
    st.plotly_chart(aplicar_estilo_grafico(fig_heatmap), use_container_width=True)

# ---------------------------------------------------------
# TAB 4: PREDICTOR
# ---------------------------------------------------------
elif selected_tab == "Predictor":
    @st.cache_resource
    def entrenar_modelos(df_in: pd.DataFrame):
        d = df_in.copy()
        d["Hora_Entrada"] = "Hora_" + d["Hora_Entrada"].astype(str)
        y = d["$Final"]

        d_model = pd.get_dummies(d, columns=["Sexo", "Procedencia", "Hora_Entrada"], drop_first=True)
        cols_sexo = [c for c in d_model.columns if c.startswith("Sexo_")]
        cols_proc = [c for c in d_model.columns if c.startswith("Procedencia_")]
        cols_hora = [c for c in d_model.columns if c.startswith("Hora_Entrada_")]
        columnas_x = ["P.Prom", "Cant."] + cols_sexo + cols_proc + cols_hora

        X_multi = d_model[columnas_x]
        X_tr_m, X_te_m, y_tr_m, y_te_m = train_test_split(X_multi, y, test_size=0.2, random_state=42)
        modelo_multi = LinearRegression().fit(X_tr_m, y_tr_m)
        r2_multi = r2_score(y_te_m, modelo_multi.predict(X_te_m))

        return {
            "modelo_multi": modelo_multi, "r2_multi": r2_multi,
            "columnas_x": columnas_x,
            "sexos": sorted(d["Sexo"].unique()),
            "procedencias": sorted(d["Procedencia"].unique()),
            "horas": sorted(d["Hora_Entrada"].astype(str).unique()),
        }

    modelos = entrenar_modelos(df)
    st.metric("Precisión R² del Modelo", f"{modelos['r2_multi']*100:.1f}%")

    colf1, colf2, colf3, colf4 = st.columns(4)
    with colf1: peso_in = st.number_input("Peso Promedio (Kg)", 20, 600, 150)
    with colf2: cant_in = st.number_input("Cantidad de animales", 1, 50, 3)
    with colf3: sexo_in = st.selectbox("Categoría", options=modelos["sexos"], format_func=etiqueta_sexo)
    with colf4: hora_in = st.selectbox("Hora de entrada", options=modelos["horas"])

    procedencia_in = st.selectbox("Procedencia Estandarizada", options=modelos["procedencias"])

    if st.button("💰 Calcular precio estimado"):
        datos_nuevos = {c: 0 for c in modelos["columnas_x"]}
        datos_nuevos["P.Prom"] = peso_in
        datos_nuevos["Cant."] = cant_in
        if f"Sexo_{sexo_in}" in datos_nuevos: datos_nuevos[f"Sexo_{sexo_in}"] = 1
        if f"Procedencia_{procedencia_in}" in datos_nuevos: datos_nuevos[f"Procedencia_{procedencia_in}"] = 1
        if f"Hora_Entrada_{hora_in}" in datos_nuevos: datos_nuevos[f"Hora_Entrada_{hora_in}"] = 1

        precio_est = modelos["modelo_multi"].predict(pd.DataFrame([datos_nuevos])[modelos["columnas_x"]])[0]
        st.success(f"Precio estimado por Kg: **${precio_est:,.0f}** | Valor estimado por animal: **${precio_est * peso_in:,.0f}**")

# ---------------------------------------------------------
# TAB 5: PRONÓSTICO
# ---------------------------------------------------------
elif selected_tab == "Pronóstico":
    st.markdown("### Pronóstico de precio semanal (Holt-Winters)")
    df_ts = df_total.copy().set_index("Fecha_TS").sort_index()
    precio_semanal = df_ts["$Final"].resample("W").mean().ffill()

    if len(precio_semanal) >= 8:
        semanas_futuro = st.slider("Semanas a pronosticar", 2, 12, 4)
        modelo_full = ExponentialSmoothing(precio_semanal, trend="add", seasonal=None).fit()
        forecast_full = modelo_full.forecast(semanas_futuro)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=precio_semanal.index, y=precio_semanal.values, mode="lines+markers", name="Histórico"))
        fig.add_trace(go.Scatter(x=forecast_full.index, y=forecast_full.values, mode="lines+markers", name="Pronóstico", line=dict(dash="dash")))
        st.plotly_chart(aplicar_estilo_grafico(fig), use_container_width=True)

# ---------------------------------------------------------
# TAB 6: PROBABILIDAD DE PUJA
# ---------------------------------------------------------
elif selected_tab == "Prob. Puja":
    st.markdown("### 🎯 Simulador Interactivo de Puja")
    cols_base = ["Cant.", "P.Prom", "$Base", "Procedencia", "Hubo_Puja"]
    df_tree = pd.get_dummies(df[cols_base].dropna(), columns=["Procedencia"], drop_first=True)

    X_tree, y_tree = df_tree.drop(columns=["Hubo_Puja"]), df_tree["Hubo_Puja"]
    modelo_arbol = DecisionTreeClassifier(max_depth=4, class_weight="balanced", random_state=42).fit(X_tree, y_tree)

    c1, c2, c3 = st.columns(3)
    p_peso = c1.slider("Peso (Kg)", 50, 600, 200)
    p_base = c2.slider("Precio Base ($/Kg)", 3000, 12000, 5000)
    p_cant = c3.slider("Cantidad", 1, 30, 5)

    input_data = pd.DataFrame(0, index=[0], columns=X_tree.columns)
    input_data.loc[0, "Cant."], input_data.loc[0, "P.Prom"], input_data.loc[0, "$Base"] = p_cant, p_peso, p_base

    prob = modelo_arbol.predict_proba(input_data)[0][1] * 100
    st.metric("Probabilidad Estimada de Recibir Puja", f"{prob:.1f}%")

# ---------------------------------------------------------
# TAB 7: COMPRADORES
# ---------------------------------------------------------
elif selected_tab == "Compradores":
    st.markdown("### Segmentación por comportamiento de compra (K-Means)")
    k_sel = st.slider("Número de perfiles (K)", 2, 5, 3)
    d = df[["P.Prom", "$Final", "Margen_Puja"]].dropna()

    scaler = StandardScaler()
    d["Perfil_ID"] = KMeans(n_clusters=k_sel, random_state=10, n_init=10).fit_predict(scaler.fit_transform(d))

    fig = px.scatter(d, x="P.Prom", y="$Final", color=d["Perfil_ID"].astype(str))
    st.plotly_chart(aplicar_estilo_grafico(fig), use_container_width=True)

# ---------------------------------------------------------
# TAB 8: MOTOR FUZZY & API (NUEVO / INTEGRADO)
# ---------------------------------------------------------
elif selected_tab == "Motor Fuzzy":
    st.markdown("### 🔤 Pruebas en Tiempo Real: Estandarizador y API Flask")
    st.caption("Verifica el funcionamiento del Diccionario Maestro, la tolerancia a errores de tipeo y la conexión HTTP.")

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        st.markdown("#### 🧪 Test Individual de Estandarización")
        texto_test = st.text_input("Ingresa un municipio (con o sin error ortográfico):", value="Sartal")
        umbral = st.slider("Umbral de similitud Fuzzy (%)", 50, 100, 75)

        if texto_test:
            res = estandarizar_procedencia(texto_test, umbral_similitud=umbral)
            st.json(res)

    with col_f2:
        st.markdown("#### 🔍 Coincidencias de Búsqueda de Dataset")
        query_busqueda = st.text_input("Buscar variante en DataFrame original:", value="Buga")
        if query_busqueda:
            terminos, canonical = construir_terminos_busqueda(query_busqueda, df_total)
            st.success(f"Categoría Estándar Asignada: **{canonical}**")
            st.write("Variantes encontradas en la columna de datos original:")
            st.code(terminos)

    st.markdown("---")
    st.markdown("#### 📡 Status de Endpoints de API Flask Activos")
    st.code("""
    POST http://localhost:5000/api/v1/estandarizar
    Header: Content-Type: application/json
    Body: {"procedencia": "Sartal"}

    GET http://localhost:5000/api/v1/buscar?q=Sartal
    """, language="bash")

# =========================================================
# 7. FOOTER INSTITUCIONAL
# =========================================================
st.markdown("""
<div class="suganorte-footer-container">
    <div class="suganorte-footer-grid">
        <div>
            <div class="footer-title">Suganorte S.A.</div>
            <p style="font-size:0.85rem; color:#E2E8F0;">Líderes en comercialización ganadera en el Suroccidente Colombiano.</p>
        </div>
        <div>
            <div class="footer-title">Información</div>
            <ul class="footer-links">
                <li>❯ Nosotros</li>
                <li>❯ Precios</li>
                <li>❯ Reglamento de Subasta</li>
            </ul>
        </div>
        <div>
            <div class="footer-title">Servicios</div>
            <ul class="footer-links">
                <li>❯ Subastas Tradicionales</li>
                <li>❯ Ventas Directas en Finca</li>
            </ul>
        </div>
        <div>
            <div class="footer-title">Contacto</div>
            <p style="font-size:0.85rem; color:#E2E8F0;">
                Km 3 Vía Zarzal - Cartago<br>
                Zarzal - Valle del Cauca<br>
                gerencia@suganorte.com.co
            </p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)
