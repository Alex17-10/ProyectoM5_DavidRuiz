import os
import pandas as pd
import streamlit as st


# ============================================================
# CONFIGURACIÓN
# ============================================================

st.set_page_config(
    page_title="Monitoreo del Modelo de Crédito",
    page_icon="📊",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MONITORING_DIR = os.path.join(
    BASE_DIR,
    "resultados_model_monitoring"
)

PREDICCIONES_FILE = os.path.join(
    BASE_DIR,
    "predicciones_api_monitoring.csv"
)

DRIFT_FILE = os.path.join(
    MONITORING_DIR,
    "data_drift_resultados.csv"
)

VARIABLES_DRIFT_FILE = os.path.join(
    MONITORING_DIR,
    "variables_con_drift.csv"
)

TEMPORAL_FILE = os.path.join(
    MONITORING_DIR,
    "analisis_temporal.csv"
)


# ============================================================
# FUNCIONES
# ============================================================

def cargar_csv(ruta):
    """Carga un archivo CSV si existe."""
    
    if not os.path.exists(ruta):
        return None

    try:
        return pd.read_csv(ruta)
    except Exception as e:
        st.error(
            f"Error al leer {os.path.basename(ruta)}: {e}"
        )
        return None


def mostrar_imagen(nombre, titulo=None):
    """Muestra una imagen del directorio de monitoring."""
    
    ruta = os.path.join(
        MONITORING_DIR,
        nombre
    )

    if os.path.exists(ruta):

        if titulo:
            st.subheader(titulo)

        st.image(
            ruta,
            width="stretch"
        )

        return True

    return False


# ============================================================
# CARGAR DATOS
# ============================================================

df_predicciones = cargar_csv(
    PREDICCIONES_FILE
)

df_drift = cargar_csv(
    DRIFT_FILE
)

df_variables_drift = cargar_csv(
    VARIABLES_DRIFT_FILE
)

df_temporal = cargar_csv(
    TEMPORAL_FILE
)


# ============================================================
# VARIABLES GENERALES
# ============================================================

total_predicciones = 0
porcentaje_incumplimiento = None
probabilidad_promedio = None
riesgo_alto = None
tiempo_promedio = None

if df_predicciones is not None:

    total_predicciones = len(
        df_predicciones
    )

    # --------------------------------------------------------
    # Porcentaje de incumplimiento
    # --------------------------------------------------------

    if "prediccion_modelo" in df_predicciones.columns:

        predicciones = pd.to_numeric(
            df_predicciones["prediccion_modelo"],
            errors="coerce"
        )

        # 0 = posible incumplimiento
        # 1 = pago a tiempo

        cantidad_incumplimiento = (
            predicciones == 0
        ).sum()

        porcentaje_incumplimiento = (
            cantidad_incumplimiento
            / total_predicciones
        ) * 100

    # --------------------------------------------------------
    # Probabilidad promedio
    # --------------------------------------------------------

    if "probabilidad_incumplimiento" in df_predicciones.columns:

        probabilidades = pd.to_numeric(
            df_predicciones[
                "probabilidad_incumplimiento"
            ],
            errors="coerce"
        )

        probabilidad_promedio = (
            probabilidades.mean() * 100
        )

    # --------------------------------------------------------
    # Riesgo alto
    # --------------------------------------------------------

    if "nivel_riesgo" in df_predicciones.columns:

        riesgo = (
            df_predicciones["nivel_riesgo"]
            .astype(str)
            .str.lower()
        )

        riesgo_alto = (
            riesgo == "alto"
        ).sum()

    # --------------------------------------------------------
    # Tiempo promedio
    # --------------------------------------------------------

    if "tiempo_prediccion_ms" in df_predicciones.columns:

        tiempos = pd.to_numeric(
            df_predicciones[
                "tiempo_prediccion_ms"
            ],
            errors="coerce"
        )

        tiempo_promedio = tiempos.mean()


# ============================================================
# ENCABEZADO
# ============================================================

st.title(
    "📊 Monitoreo del Modelo de Crédito"
)

st.write(
    "Dashboard de seguimiento del modelo de "
    "predicción de incumplimiento de créditos."
)

st.write(
    "Permite consultar las predicciones generadas "
    "por la API, analizar el nivel de riesgo y "
    "revisar posibles cambios en la distribución "
    "de los datos mediante Data Drift."
)


# ============================================================
# NAVEGACIÓN
# ============================================================

st.sidebar.title(
    "Navegación"
)

seccion = st.sidebar.radio(
    "Selecciona una sección:",
    [
        "Resumen",
        "Predicciones",
        "Data Drift",
        "Análisis temporal"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info(
    "Los datos mostrados corresponden a los "
    "resultados generados por el proceso de "
    "monitoreo del modelo."
)


# ============================================================
# 1. RESUMEN
# ============================================================

if seccion == "Resumen":

    st.header(
        "Resumen del monitoreo"
    )

    if df_predicciones is None:

        st.error(
            "No se encontró el archivo "
            "`predicciones_api_monitoring.csv`."
        )

    else:

        # ----------------------------------------------------
        # MÉTRICAS
        # ----------------------------------------------------

        col1, col2, col3, col4, col5 = st.columns(5)

        col1.metric(
            "Total predicciones",
            f"{total_predicciones:,}"
        )

        if porcentaje_incumplimiento is not None:

            col2.metric(
                "% Incumplimiento",
                f"{porcentaje_incumplimiento:.2f}%"
            )

        else:

            col2.metric(
                "% Incumplimiento",
                "N/D"
            )

        if probabilidad_promedio is not None:

            col3.metric(
                "Probabilidad promedio",
                f"{probabilidad_promedio:.2f}%"
            )

        else:

            col3.metric(
                "Probabilidad promedio",
                "N/D"
            )

        if riesgo_alto is not None:

            col4.metric(
                "Riesgo alto",
                f"{riesgo_alto:,}"
            )

        else:

            col4.metric(
                "Riesgo alto",
                "N/D"
            )

        if tiempo_promedio is not None:

            col5.metric(
                "Tiempo promedio",
                f"{tiempo_promedio:.2f} ms"
            )

        else:

            col5.metric(
                "Tiempo promedio",
                "N/D"
            )

        st.markdown("---")

        # ----------------------------------------------------
        # GRÁFICOS
        # ----------------------------------------------------

        st.header(
            "Distribuciones del modelo"
        )

        col1, col2 = st.columns(2)

        with col1:

            mostrar_imagen(
                "distribucion_predicciones.png",
                "Distribución de predicciones"
            )

        with col2:

            mostrar_imagen(
                "distribucion_probabilidad_incumplimiento.png",
                "Probabilidad de incumplimiento"
            )

        mostrar_imagen(
            "distribucion_niveles_riesgo.png",
            "Distribución de niveles de riesgo"
        )


# ============================================================
# 2. PREDICCIONES
# ============================================================

elif seccion == "Predicciones":

    st.header(
        "Predicciones generadas por la API"
    )

    if df_predicciones is None:

        st.error(
            "No se encontró "
            "`predicciones_api_monitoring.csv`."
        )

    else:

        st.write(
            f"Se encontraron "
            f"**{len(df_predicciones):,} predicciones**."
        )

        # ----------------------------------------------------
        # FILTRO DE RIESGO
        # ----------------------------------------------------

        df_filtrado = df_predicciones.copy()

        if "nivel_riesgo" in df_filtrado.columns:

            niveles = sorted(
                df_filtrado[
                    "nivel_riesgo"
                ]
                .dropna()
                .astype(str)
                .unique()
            )

            niveles_seleccionados = st.multiselect(
                "Filtrar por nivel de riesgo:",
                niveles,
                default=niveles
            )

            df_filtrado = df_filtrado[
                df_filtrado[
                    "nivel_riesgo"
                ]
                .astype(str)
                .isin(niveles_seleccionados)
            ]

        # ----------------------------------------------------
        # FILTRO DE PROBABILIDAD
        # ----------------------------------------------------

        if (
            "probabilidad_incumplimiento"
            in df_filtrado.columns
        ):

            probabilidades = pd.to_numeric(
                df_predicciones[
                    "probabilidad_incumplimiento"
                ],
                errors="coerce"
            )

            minimo = float(
                probabilidades.min()
            )

            maximo = float(
                probabilidades.max()
            )

            if minimo < maximo:

                rango = st.slider(
                    "Probabilidad de incumplimiento:",
                    min_value=minimo,
                    max_value=maximo,
                    value=(minimo, maximo)
                )

                probabilidades_filtradas = pd.to_numeric(
                    df_filtrado[
                        "probabilidad_incumplimiento"
                    ],
                    errors="coerce"
                )

                df_filtrado = df_filtrado[
                    (
                        probabilidades_filtradas
                        >= rango[0]
                    )
                    &
                    (
                        probabilidades_filtradas
                        <= rango[1]
                    )
                ]

        st.write(
            f"Registros mostrados: "
            f"**{len(df_filtrado):,}**"
        )

        st.dataframe(
            df_filtrado,
            width="stretch",
            hide_index=True
        )


# ============================================================
# 3. DATA DRIFT
# ============================================================

elif seccion == "Data Drift":

    st.header(
        "📈 Data Drift"
    )

    st.write(
        "Comparación entre los datos de referencia "
        "y los datos recientes utilizados para "
        "generar predicciones."
    )

    if df_drift is None:

        st.error(
            "No se encontró "
            "`data_drift_resultados.csv`."
        )

    else:

        # ----------------------------------------------------
        # VARIABLES MONITORIZADAS
        # ----------------------------------------------------

        total_variables = len(
            df_drift
        )

        # ----------------------------------------------------
        # VARIABLES CON DRIFT
        # ----------------------------------------------------

        if df_variables_drift is not None:

            variables_con_drift = len(
                df_variables_drift
            )

        elif "drift_detectado" in df_drift.columns:

            variables_con_drift = int(
                df_drift[
                    "drift_detectado"
                ]
                .fillna(False)
                .astype(bool)
                .sum()
            )

        else:

            variables_con_drift = 0

        # ----------------------------------------------------
        # PORCENTAJE
        # ----------------------------------------------------

        if total_variables > 0:

            porcentaje_drift = (
                variables_con_drift
                / total_variables
            ) * 100

        else:

            porcentaje_drift = 0

        # ----------------------------------------------------
        # MÉTRICAS
        # ----------------------------------------------------

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Variables monitorizadas",
            total_variables
        )

        col2.metric(
            "Variables con drift",
            variables_con_drift
        )

        col3.metric(
            "% con drift",
            f"{porcentaje_drift:.2f}%"
        )

        st.markdown("---")

        # ----------------------------------------------------
        # TABLA PRINCIPAL
        # ----------------------------------------------------

        st.subheader(
            "Resultados de Data Drift"
        )

        st.dataframe(
            df_drift,
            width="stretch",
            hide_index=True
        )

        # ----------------------------------------------------
        # VARIABLES CON DRIFT
        # ----------------------------------------------------

        if df_variables_drift is not None:

            st.subheader(
                "Variables con señales de drift"
            )

            st.dataframe(
                df_variables_drift,
                width="stretch",
                hide_index=True
            )

        # ----------------------------------------------------
        # GRÁFICOS
        # ----------------------------------------------------

        st.subheader(
            "Gráficos de Data Drift"
        )

        drift_graphics_dir = os.path.join(
            MONITORING_DIR,
            "data_drift_graficos"
        )

        if os.path.exists(
            drift_graphics_dir
        ):

            archivos = sorted(
                [
                    archivo
                    for archivo in os.listdir(
                        drift_graphics_dir
                    )
                    if archivo.lower().endswith(
                        (
                            ".png",
                            ".jpg",
                            ".jpeg"
                        )
                    )
                ]
            )

            if archivos:

                for archivo in archivos:

                    st.image(
                        os.path.join(
                            drift_graphics_dir,
                            archivo
                        ),
                        caption=archivo,
                        width="stretch"
                    )

            else:

                st.info(
                    "No se encontraron gráficos "
                    "de Data Drift."
                )

        else:

            st.info(
                "No se encontró la carpeta "
                "`data_drift_graficos`."
            )


# ============================================================
# 4. ANÁLISIS TEMPORAL
# ============================================================

elif seccion == "Análisis temporal":

    st.header(
        "📅 Análisis temporal"
    )

    if df_temporal is None:

        st.error(
            "No se encontró "
            "`analisis_temporal.csv`."
        )

    else:

        st.subheader(
            "Evolución temporal de las predicciones"
        )

        st.dataframe(
            df_temporal,
            width="stretch",
            hide_index=True
        )

        # ----------------------------------------------------
        # DETECTAR COLUMNA TEMPORAL
        # ----------------------------------------------------

        columnas_fecha = []

        for columna in df_temporal.columns:

            nombre = columna.lower()

            if (
                "fecha" in nombre
                or "periodo" in nombre
                or "mes" in nombre
                or "date" in nombre
            ):

                columnas_fecha.append(
                    columna
                )

        if columnas_fecha:

            columna_fecha = (
                columnas_fecha[0]
            )

            df_grafico = (
                df_temporal.copy()
            )

            df_grafico[
                columna_fecha
            ] = pd.to_datetime(
                df_grafico[
                    columna_fecha
                ],
                errors="coerce"
            )

            df_grafico = (
                df_grafico
                .dropna(
                    subset=[
                        columna_fecha
                    ]
                )
            )

            if not df_grafico.empty:

                df_grafico = (
                    df_grafico
                    .set_index(
                        columna_fecha
                    )
                )

                columnas_numericas = (
                    df_grafico
                    .select_dtypes(
                        include="number"
                    )
                    .columns
                )

                if len(
                    columnas_numericas
                ) > 0:

                    st.subheader(
                        "Evolución de variables"
                    )

                    st.line_chart(
                        df_grafico[
                            columnas_numericas
                        ]
                    )

        # ----------------------------------------------------
        # GRÁFICO DEL MONITORING
        # ----------------------------------------------------

        mostrar_imagen(
            "analisis_temporal_predicciones.png",
            "Análisis temporal generado por el monitoreo"
        )


# ============================================================
# PIE DE PÁGINA
# ============================================================

st.markdown("---")

st.caption(
    "Sistema de monitoreo del modelo de "
    "predicción de incumplimiento de créditos."
)