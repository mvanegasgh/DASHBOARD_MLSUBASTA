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
# RECURSOS DE MARCA (URLs DIRECTAS / FALLBACK)
# =========================================================
# Si prefieres archivo local, usa: LOGO_COLOR = "logo2x-suganorte.png"
LOGO_COLOR_URL = "https://raw.githubusercontent.com/suganorte/assets/main/logo2x-suganorte.png" 
LOGO_WHITE_URL = "https://raw.githubusercontent.com/suganorte/assets/main/logo2x-suganorte-white.png"

# =========================================================
# ESTILOS CSS CORREGIDOS (INPUTS Y TEMAS LIMPIOS)
# =========================================================
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    /* BASE DE LA APP */
    html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {{
        background-color: #F8FAFC !important;
        color: #0F172A !important;
        font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
    }}

    /* BANNER INSTITUCIONAL OFICIAL */
    .suganorte-header-container {{
        background: #0B2265;
        border-radius: 12px;
        overflow: hidden;
        margin-bottom: 24px;
        box-shadow: 0 4px 15px rgba(11, 34, 101, 0.15);
    }}
    .tricolor-stripe {{
        height: 6px;
        background: linear-gradient(90deg, #FFD100 0% 33%, #008037 33% 66%, #E11D48 66% 100%);
        width: 100%;
    }}
    .suganorte-banner-body {{
        padding: 24px 32px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 20px;
    }}
    .suganorte-title {{
        color: #FFFFFF !important;
        margin: 0 !important;
        font-size: 1.75rem !important;
        font-weight: 800 !important;
        letter-spacing: -0.02em;
    }}
    .suganorte-subtitle {{
        color: #FFD100 !important;
        margin-top: 6px !important;
        margin-bottom: 0 !important;
        font-size: 0.95rem !important;
        font-weight: 600 !important;
    }}

    /* BARRA LATERAL Y FILTROS (LIMPIEZA DE COLORES CREMA/MARRÓN) */
    section[data-testid="stSidebar"], [data-testid="stSidebarContent"] {{
        background-color: #FFFFFF !important;
        border-right: 1px solid #E2E8F0 !important;
    }}
    
    /* INPUTS Y SELECTORES (REGLAS DE COLOR INSTITUCIONAL) */
    div[data-baseweb="select"] > div, div[data-baseweb="input"] > div {{
        background-color: #F1F5F9 !important;
        border-color: #CBD5E1 !important;
        color: #0F172A !important;
        border-radius: 8px !important;
    }}
    
    /* CHIPS/ETIQUETAS DEL MULTISELECT */
    span[data-baseweb="tag"] {{
        background-color: #0B2265 !important;
        border-radius: 6px !important;
    }}
    span[data-baseweb="tag"] span {{
        color: #FFFFFF !important;
        font-weight: 600 !important;
    }}

    /* METRICAS Y KPIS */
    div[data-testid="stMetric"] {{
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-top: 4px solid #008037 !important;
        border-radius: 10px !important;
        padding: 16px 20px !important;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.02) !important;
    }}
    div[data-testid="stMetric"] label {{
        color: #64748B !important;
        font-size: 0.8rem !important;
        font-weight: 700 !important;
        text-transform: uppercase;
    }}
    div[data-testid="stMetricValue"] {{
        color: #0B2265 !important;
        font-size: 1.8rem !important;
        font-weight: 800 !important;
    }}

    /* BOTONES */
    .stButton>button {{
        background-color: #008037 !important;
        color: #FFFFFF !important;
        border-radius: 8px !important;
        border: none !important;
        font-weight: 700 !important;
        padding: 10px 24px !important;
    }}
    .stButton>button:hover {{
        background-color: #006028 !important;
    }}
</style>
""", unsafe_allow_html=True)

# (El resto del script mantiene la estructura de datos, sidebar y tabs previamente armados)
