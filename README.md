# Dashboard de Analítica Predictiva — Subasta Ganadera Suganorte

Dashboard interactivo en Streamlit construido a partir del notebook
`Analitica_Predictiva_Subasta_Ganadera`. Incluye:

- **Resumen**: KPIs generales, evolución de precios, volumen por categoría/procedencia.
- **Exploración (EDA)**: distribución de precios, peso vs. precio, boxplots por categoría.
- **Correlación**: correlación de variables contra el precio final + mapa de calor.
- **Regresión y Predictor**: comparación regresión simple vs. múltiple, impacto económico
  por variable, y un **simulador interactivo** para estimar el precio por Kg de un lote
  (peso, cantidad, sexo, procedencia, hora de entrada).
- **Pronóstico**: series de tiempo semanales con Holt-Winters y métricas de error (MAE/RMSE/MAPE).
- **Prob. de Puja**: árbol de decisión que estima la probabilidad de que un lote reciba puja.
- **Perfiles de Compradores**: segmentación K-Means (perfiles estratégicos de mercado).

Los filtros de fecha, categoría (Sexo) y procedencia en la barra lateral afectan a
todas las pestañas.

## Archivos

- `app.py` — aplicación Streamlit.
- `data.csv` — dataset limpio de la subasta (mismo archivo subido).
- `requirements.txt` — dependencias.

## Ejecutar localmente

```bash
pip install -r requirements.txt
streamlit run app.py
```

Abre `http://localhost:8501`.

## Desplegar en Streamlit Community Cloud (igual que el ejemplo de referencia)

1. Sube esta carpeta (`app.py`, `data.csv`, `requirements.txt`) a un repositorio de GitHub.
2. Entra a [share.streamlit.io](https://share.streamlit.io) con tu cuenta de GitHub.
3. Click en **"New app"**, selecciona el repo, la rama, y como archivo principal `app.py`.
4. Click en **Deploy**. En 1–2 minutos tendrás una URL pública tipo
   `https://tu-app.streamlit.app`, igual que el ejemplo que compartiste.

> Nota: no pude abrir la URL de referencia (`subadatos-centralganadera.streamlit.app`)
> porque el sitio bloquea el acceso automatizado (robots.txt). Este dashboard se
> construyó replicando y ampliando todo el análisis de tu notebook (regresión,
> series de tiempo, árbol de decisión y clustering) con una estructura típica de
> tablero de subasta ganadera. Si me compartes una captura de pantalla o describes
> secciones específicas del dashboard de referencia, con gusto ajusto el diseño o
> el contenido para que se parezca más.
