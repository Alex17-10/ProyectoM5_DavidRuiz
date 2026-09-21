import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import ks_2samp, chi2_contingency
from scipy.spatial.distance import jensenshannon


# ============================================================
# RUTAS PRINCIPALES DEL PROYECTO
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

RUTA_BASE_REFERENCIA = os.path.join(
    BASE_DIR,
    "Base_de_datos_preparada.csv"
)

RUTA_PREDICCIONES = os.path.join(
    BASE_DIR,
    "predicciones_api_monitoring.csv"
)

CARPETA_RESULTADOS = os.path.join(
    BASE_DIR,
    "resultados_model_monitoring"
)

CARPETA_DRIFT = os.path.join(
    CARPETA_RESULTADOS,
    "data_drift_graficos"
)


# ============================================================
# CREAR CARPETAS
# ============================================================

os.makedirs(
    CARPETA_RESULTADOS,
    exist_ok=True
)

os.makedirs(
    CARPETA_DRIFT,
    exist_ok=True
)


# ============================================================
# RUTAS DE SALIDA
# ============================================================

RUTA_METRICAS = os.path.join(
    CARPETA_RESULTADOS,
    "metricas_monitoring.csv"
)

RUTA_DISTRIBUCION_PREDICCIONES = os.path.join(
    CARPETA_RESULTADOS,
    "distribucion_predicciones.csv"
)

RUTA_PREDICCIONES_VALIDAS = os.path.join(
    CARPETA_RESULTADOS,
    "predicciones_validas_monitoring.csv"
)

RUTA_RESUMEN_RIESGO = os.path.join(
    CARPETA_RESULTADOS,
    "resumen_niveles_riesgo.csv"
)

RUTA_GRAFICO_PREDICCIONES = os.path.join(
    CARPETA_RESULTADOS,
    "distribucion_predicciones.png"
)

RUTA_GRAFICO_PROBABILIDADES = os.path.join(
    CARPETA_RESULTADOS,
    "distribucion_probabilidad_incumplimiento.png"
)

RUTA_GRAFICO_RIESGO = os.path.join(
    CARPETA_RESULTADOS,
    "distribucion_niveles_riesgo.png"
)

RUTA_ALERTAS = os.path.join(
    CARPETA_RESULTADOS,
    "alertas_monitoring.txt"
)

RUTA_DRIFT = os.path.join(
    CARPETA_RESULTADOS,
    "data_drift_resultados.csv"
)

RUTA_DRIFT_ALERTAS = os.path.join(
    CARPETA_RESULTADOS,
    "data_drift_alertas.txt"
)

RUTA_VARIABLES_DRIFT = os.path.join(
    CARPETA_RESULTADOS,
    "variables_con_drift.csv"
)

RUTA_RESUMEN_TEMPORAL = os.path.join(
    CARPETA_RESULTADOS,
    "analisis_temporal.csv"
)


# ============================================================
# INICIO DEL MONITOREO
# ============================================================

print("Inicio del model monitoring")
print()

print(
    f"Buscando base de referencia en: "
    f"{RUTA_BASE_REFERENCIA}"
)

print(
    f"Buscando predicciones en: "
    f"{RUTA_PREDICCIONES}"
)

print()


# ============================================================
# VERIFICAR ARCHIVOS
# ============================================================

if not os.path.exists(RUTA_BASE_REFERENCIA):

    print(
        "No se encontró la base de datos de referencia."
    )

    print(
        "Debe existir el archivo "
        "'Base_de_datos_preparada.csv'."
    )

    raise SystemExit


if not os.path.exists(RUTA_PREDICCIONES):

    print(
        "No se encontró el archivo de predicciones de la API."
    )

    print(
        "Ejecuta primero una o varias predicciones "
        "desde /predict o /predict/batch."
    )

    raise SystemExit


# ============================================================
# CARGAR BASE DE REFERENCIA
# ============================================================

try:

    df_referencia = pd.read_csv(
        RUTA_BASE_REFERENCIA
    )

except Exception as error:

    print(
        f"No fue posible cargar la base de referencia: "
        f"{error}"
    )

    raise SystemExit


# ============================================================
# CARGAR PREDICCIONES
# ============================================================

try:

    df = pd.read_csv(
        RUTA_PREDICCIONES
    )

except Exception as error:

    print(
        f"No fue posible cargar el archivo de predicciones: "
        f"{error}"
    )

    raise SystemExit


print(
    f"Base de referencia cargada correctamente: "
    f"{len(df_referencia)} registros"
)

print(
    f"Archivo de predicciones cargado correctamente: "
    f"{len(df)} registros"
)

print()


# ============================================================
# VERIFICAR COLUMNAS ESENCIALES DE PREDICCIONES
# ============================================================

columnas_esenciales = [
    "prediccion_modelo",
    "resultado_prediccion",
    "probabilidad_incumplimiento",
    "probabilidad_pago_atiempo"
]

columnas_faltantes = [
    columna
    for columna in columnas_esenciales
    if columna not in df.columns
]

if columnas_faltantes:

    print(
        "El archivo no contiene las columnas necesarias "
        "para el monitoreo."
    )

    print(
        "Columnas faltantes:",
        columnas_faltantes
    )

    print()

    print(
        "Esto normalmente ocurre cuando el CSV fue generado "
        "con una versión anterior de la API."
    )

    raise SystemExit


# ============================================================
# CONVERSIÓN DE COLUMNAS NUMÉRICAS
# ============================================================

columnas_numericas_api = [
    "prediccion_modelo",
    "probabilidad_incumplimiento",
    "probabilidad_pago_atiempo",
    "tiempo_prediccion_ms"
]

for columna in columnas_numericas_api:

    if columna in df.columns:

        df[columna] = pd.to_numeric(
            df[columna],
            errors="coerce"
        )


# ============================================================
# IDENTIFICAR PREDICCIONES VÁLIDAS
# ============================================================

df_validas = df[
    df["prediccion_modelo"].isin([0, 1])
].copy()


df_validas = df_validas[
    df_validas[
        "probabilidad_incumplimiento"
    ].between(
        0,
        1,
        inclusive="both"
    )
]


df_validas = df_validas[
    df_validas[
        "probabilidad_pago_atiempo"
    ].between(
        0,
        1,
        inclusive="both"
    )
].copy()


# ============================================================
# VALIDAR CONSISTENCIA DE PROBABILIDADES
# ============================================================

df_validas["suma_probabilidades"] = (
    df_validas["probabilidad_incumplimiento"]
    +
    df_validas["probabilidad_pago_atiempo"]
)


df_validas["probabilidades_consistentes"] = (
    df_validas["suma_probabilidades"].between(
        0.99,
        1.01,
        inclusive="both"
    )
)


df_validas = df_validas[
    df_validas["probabilidades_consistentes"]
].copy()


# ============================================================
# GUARDAR PREDICCIONES VÁLIDAS
# ============================================================

df_validas.to_csv(
    RUTA_PREDICCIONES_VALIDAS,
    index=False
)


# ============================================================
# CANTIDAD DE REGISTROS
# ============================================================

total_registros = len(df)

total_validos = len(df_validas)

total_invalidos = (
    total_registros -
    total_validos
)


if total_registros == 0:

    print(
        "El archivo no contiene registros."
    )

    raise SystemExit


if total_validos == 0:

    print(
        "No existen predicciones válidas para monitorear."
    )

    raise SystemExit


# ============================================================
# DISTRIBUCIÓN DE LAS PREDICCIONES
# ============================================================

cantidad_incumplimiento = int(
    (
        df_validas["prediccion_modelo"] == 0
    ).sum()
)


cantidad_pago_atiempo = int(
    (
        df_validas["prediccion_modelo"] == 1
    ).sum()
)


porcentaje_incumplimiento = (
    cantidad_incumplimiento /
    total_validos
) * 100


porcentaje_pago_atiempo = (
    cantidad_pago_atiempo /
    total_validos
) * 100


distribucion_predicciones = pd.DataFrame(
    {
        "prediccion_modelo": [
            0,
            1
        ],

        "resultado_prediccion": [
            "Posible incumplimiento",
            "Pago a tiempo"
        ],

        "cantidad": [
            cantidad_incumplimiento,
            cantidad_pago_atiempo
        ],

        "porcentaje": [
            porcentaje_incumplimiento,
            porcentaje_pago_atiempo
        ]
    }
)


distribucion_predicciones.to_csv(
    RUTA_DISTRIBUCION_PREDICCIONES,
    index=False
)


# ============================================================
# ESTADÍSTICAS DE PROBABILIDADES
# ============================================================

promedio_probabilidad_incumplimiento = (
    df_validas[
        "probabilidad_incumplimiento"
    ].mean()
)


promedio_probabilidad_pago_atiempo = (
    df_validas[
        "probabilidad_pago_atiempo"
    ].mean()
)


mediana_probabilidad_incumplimiento = (
    df_validas[
        "probabilidad_incumplimiento"
    ].median()
)


desviacion_probabilidad_incumplimiento = (
    df_validas[
        "probabilidad_incumplimiento"
    ].std()
)


minima_probabilidad_incumplimiento = (
    df_validas[
        "probabilidad_incumplimiento"
    ].min()
)


maxima_probabilidad_incumplimiento = (
    df_validas[
        "probabilidad_incumplimiento"
    ].max()
)


# ============================================================
# MONITOREO DE NIVELES DE RIESGO
# ============================================================

if "nivel_riesgo" in df_validas.columns:

    df_validas["nivel_riesgo"] = (
        df_validas["nivel_riesgo"]
        .fillna("No disponible")
    )

else:

    def clasificar_riesgo(probabilidad):

        if pd.isna(probabilidad):

            return "No disponible"

        if probabilidad >= 0.70:

            return "Alto"

        if probabilidad >= 0.40:

            return "Medio"

        return "Bajo"


    df_validas["nivel_riesgo"] = (
        df_validas[
            "probabilidad_incumplimiento"
        ].apply(
            clasificar_riesgo
        )
    )


orden_riesgo = [
    "Bajo",
    "Medio",
    "Alto",
    "No disponible"
]


resumen_riesgo = (
    df_validas["nivel_riesgo"]
    .value_counts()
    .reindex(
        orden_riesgo,
        fill_value=0
    )
    .reset_index()
)


resumen_riesgo.columns = [
    "nivel_riesgo",
    "cantidad"
]


resumen_riesgo["porcentaje"] = (
    resumen_riesgo["cantidad"] /
    total_validos
) * 100


resumen_riesgo.to_csv(
    RUTA_RESUMEN_RIESGO,
    index=False
)


# ============================================================
# PORCENTAJE DE RIESGO ALTO
# ============================================================

cantidad_riesgo_alto = int(
    (
        df_validas["nivel_riesgo"] == "Alto"
    ).sum()
)


porcentaje_riesgo_alto = (
    cantidad_riesgo_alto /
    total_validos
) * 100


# ============================================================
# MONITOREO DEL TIEMPO DE PREDICCIÓN
# ============================================================

if "tiempo_prediccion_ms" in df_validas.columns:

    tiempos = df_validas[
        "tiempo_prediccion_ms"
    ].dropna()

    if len(tiempos) > 0:

        promedio_tiempo_ms = tiempos.mean()
        mediana_tiempo_ms = tiempos.median()
        minimo_tiempo_ms = tiempos.min()
        maximo_tiempo_ms = tiempos.max()

    else:

        promedio_tiempo_ms = np.nan
        mediana_tiempo_ms = np.nan
        minimo_tiempo_ms = np.nan
        maximo_tiempo_ms = np.nan

else:

    promedio_tiempo_ms = np.nan
    mediana_tiempo_ms = np.nan
    minimo_tiempo_ms = np.nan
    maximo_tiempo_ms = np.nan


# ============================================================
# FUNCIONES DE DATA DRIFT
# ============================================================

def calcular_psi(
    referencia,
    actual,
    numero_bins=10
):

    referencia = pd.Series(
        referencia
    ).dropna()

    actual = pd.Series(
        actual
    ).dropna()

    if len(referencia) == 0 or len(actual) == 0:

        return np.nan

    try:

        cuantiles = np.unique(
            np.percentile(
                referencia,
                np.linspace(
                    0,
                    100,
                    numero_bins + 1
                )
            )
        )

        if len(cuantiles) < 3:

            return 0.0

        referencia_bins = pd.cut(
            referencia,
            bins=cuantiles,
            include_lowest=True,
            duplicates="drop"
        )

        actual_bins = pd.cut(
            actual,
            bins=cuantiles,
            include_lowest=True,
            duplicates="drop"
        )

        proporcion_referencia = (
            referencia_bins
            .value_counts(
                normalize=True,
                sort=False
            )
        )

        proporcion_actual = (
            actual_bins
            .value_counts(
                normalize=True,
                sort=False
            )
        )

        proporcion_referencia, proporcion_actual = (
            proporcion_referencia.align(
                proporcion_actual,
                fill_value=0
            )
        )

        epsilon = 0.0001

        proporcion_referencia = (
            proporcion_referencia
            .clip(lower=epsilon)
        )

        proporcion_actual = (
            proporcion_actual
            .clip(lower=epsilon)
        )

        psi = np.sum(
            (
                proporcion_actual -
                proporcion_referencia
            )
            *
            np.log(
                proporcion_actual /
                proporcion_referencia
            )
        )

        return float(psi)

    except Exception:

        return np.nan


def calcular_psi_categorico(
    referencia,
    actual
):

    referencia = pd.Series(
        referencia
    ).fillna(
        "MISSING"
    ).astype(str)

    actual = pd.Series(
        actual
    ).fillna(
        "MISSING"
    ).astype(str
    )

    categorias = sorted(
        set(
            referencia.unique()
        )
        |
        set(
            actual.unique()
        )
    )

    if len(categorias) == 0:

        return np.nan

    referencia_frecuencias = (
        referencia.value_counts(
            normalize=True
        )
        .reindex(
            categorias,
            fill_value=0
        )
    )

    actual_frecuencias = (
        actual.value_counts(
            normalize=True
        )
        .reindex(
            categorias,
            fill_value=0
        )
    )

    epsilon = 0.0001

    referencia_frecuencias = (
        referencia_frecuencias
        .clip(lower=epsilon)
    )

    actual_frecuencias = (
        actual_frecuencias
        .clip(lower=epsilon)
    )

    psi = np.sum(
        (
            actual_frecuencias -
            referencia_frecuencias
        )
        *
        np.log(
            actual_frecuencias /
            referencia_frecuencias
        )
    )

    return float(psi)


def calcular_js_numerico(
    referencia,
    actual,
    numero_bins=20
):

    referencia = pd.Series(
        referencia
    ).dropna()

    actual = pd.Series(
        actual
    ).dropna()

    if len(referencia) == 0 or len(actual) == 0:

        return np.nan

    try:

        minimo = min(
            referencia.min(),
            actual.min()
        )

        maximo = max(
            referencia.max(),
            actual.max()
        )

        if minimo == maximo:

            return 0.0

        bins = np.linspace(
            minimo,
            maximo,
            numero_bins + 1
        )

        hist_referencia, _ = np.histogram(
            referencia,
            bins=bins,
            density=False
        )

        hist_actual, _ = np.histogram(
            actual,
            bins=bins,
            density=False
        )

        epsilon = 0.0001

        hist_referencia = (
            hist_referencia.astype(float)
            + epsilon
        )

        hist_actual = (
            hist_actual.astype(float)
            + epsilon
        )

        hist_referencia = (
            hist_referencia /
            hist_referencia.sum()
        )

        hist_actual = (
            hist_actual /
            hist_actual.sum()
        )

        distancia = jensenshannon(
            hist_referencia,
            hist_actual,
            base=2
        )

        return float(distancia)

    except Exception:

        return np.nan


def calcular_js_categorico(
    referencia,
    actual
):

    referencia = pd.Series(
        referencia
    ).fillna(
        "MISSING"
    ).astype(str)

    actual = pd.Series(
        actual
    ).fillna(
        "MISSING"
    ).astype(str)

    categorias = sorted(
        set(
            referencia.unique()
        )
        |
        set(
            actual.unique()
        )
    )

    if len(categorias) == 0:

        return np.nan

    frecuencia_referencia = (
        referencia.value_counts(
            normalize=True
        )
        .reindex(
            categorias,
            fill_value=0
        )
        .values
    )

    frecuencia_actual = (
        actual.value_counts(
            normalize=True
        )
        .reindex(
            categorias,
            fill_value=0
        )
        .values
    )

    epsilon = 0.0001

    frecuencia_referencia = (
        frecuencia_referencia +
        epsilon
    )

    frecuencia_actual = (
        frecuencia_actual +
        epsilon
    )

    frecuencia_referencia = (
        frecuencia_referencia /
        frecuencia_referencia.sum()
    )

    frecuencia_actual = (
        frecuencia_actual /
        frecuencia_actual.sum()
    )

    distancia = jensenshannon(
        frecuencia_referencia,
        frecuencia_actual,
        base=2
    )

    return float(distancia)


def calcular_chi_cuadrado(
    referencia,
    actual
):

    referencia = pd.Series(
        referencia
    ).fillna(
        "MISSING"
    ).astype(str)

    actual = pd.Series(
        actual
    ).fillna(
        "MISSING"
    ).astype(str)

    categorias = sorted(
        set(
            referencia.unique()
        )
        |
        set(
            actual.unique()
        )
    )

    if len(categorias) < 2:

        return np.nan, np.nan

    tabla_referencia = (
        referencia.value_counts()
        .reindex(
            categorias,
            fill_value=0
        )
    )

    tabla_actual = (
        actual.value_counts()
        .reindex(
            categorias,
            fill_value=0
        )
    )

    tabla_contingencia = np.array(
        [
            tabla_referencia.values,
            tabla_actual.values
        ]
    )

    try:

        chi2, p_value, _, _ = (
            chi2_contingency(
                tabla_contingencia
            )
        )

        return (
            float(chi2),
            float(p_value)
        )

    except Exception:

        return np.nan, np.nan


# ============================================================
# IDENTIFICAR COLUMNAS PARA DATA DRIFT
# ============================================================

columnas_excluidas_drift = [
    "id_prediccion",
    "fecha_prediccion_api",
    "prediccion_modelo",
    "resultado_prediccion",
    "probabilidad_incumplimiento",
    "probabilidad_pago_atiempo",
    "nivel_riesgo",
    "tiempo_prediccion_ms",
    "Pago_atiempo"
]


columnas_comunes = [
    columna
    for columna in df_referencia.columns
    if columna in df_validas.columns
    and columna not in columnas_excluidas_drift
]


if len(columnas_comunes) == 0:

    print(
        "No existen columnas comunes entre la base "
        "de referencia y las predicciones."
    )

    raise SystemExit


# ============================================================
# DETECTAR TIPOS DE VARIABLES
# ============================================================

columnas_numericas_drift = []

columnas_categoricas_drift = []

columnas_fecha_drift = []


for columna in columnas_comunes:

    serie_referencia = (
        df_referencia[columna]
    )

    serie_actual = (
        df_validas[columna]
    )

    if (
        pd.api.types.is_datetime64_any_dtype(
            serie_referencia
        )
        or
        "fecha" in columna.lower()
    ):

        try:

            pd.to_datetime(
                serie_referencia,
                errors="raise"
            )

            pd.to_datetime(
                serie_actual,
                errors="raise"
            )

            columnas_fecha_drift.append(
                columna
            )

            continue

        except Exception:

            pass

    if (
        pd.api.types.is_numeric_dtype(
            serie_referencia
        )
        and
        pd.api.types.is_numeric_dtype(
            serie_actual
        )
    ):

        columnas_numericas_drift.append(
            columna
        )

    else:

        columnas_categoricas_drift.append(
            columna
        )


# ============================================================
# CALCULAR DATA DRIFT
# ============================================================

resultados_drift = []


for columna in columnas_numericas_drift:

    referencia = pd.to_numeric(
        df_referencia[columna],
        errors="coerce"
    ).dropna()

    actual = pd.to_numeric(
        df_validas[columna],
        errors="coerce"
    ).dropna()

    if len(referencia) < 2 or len(actual) < 2:

        continue

    try:

        ks_statistic, ks_p_value = ks_2samp(
            referencia,
            actual
        )

    except Exception:

        ks_statistic = np.nan
        ks_p_value = np.nan


    psi = calcular_psi(
        referencia,
        actual
    )


    js = calcular_js_numerico(
        referencia,
        actual
    )


    drift_ks = (
        not pd.isna(ks_p_value)
        and ks_p_value < 0.05
    )


    drift_psi = (
        not pd.isna(psi)
        and psi >= 0.25
    )


    drift_js = (
        not pd.isna(js)
        and js >= 0.10
    )


    drift_detectado = (
        drift_ks
        or
        drift_psi
        or
        drift_js
    )


    if (
        not pd.isna(psi)
        and psi >= 0.25
    ):

        nivel_drift = "Alto"

    elif (
        not pd.isna(psi)
        and psi >= 0.10
    ):

        nivel_drift = "Moderado"

    elif drift_detectado:

        nivel_drift = "Detectado"

    else:

        nivel_drift = "Estable"


    resultados_drift.append(
        {
            "variable": columna,
            "tipo_variable": "Numerica",
            "ks_statistic": ks_statistic,
            "ks_p_value": ks_p_value,
            "psi": psi,
            "jensen_shannon": js,
            "chi2": np.nan,
            "chi2_p_value": np.nan,
            "drift_ks": drift_ks,
            "drift_psi": drift_psi,
            "drift_js": drift_js,
            "drift_chi2": False,
            "drift_detectado": drift_detectado,
            "nivel_drift": nivel_drift,
            "registros_referencia": len(referencia),
            "registros_actuales": len(actual)
        }
    )


for columna in columnas_categoricas_drift:

    referencia = (
        df_referencia[columna]
        .fillna("MISSING")
        .astype(str)
    )

    actual = (
        df_validas[columna]
        .fillna("MISSING")
        .astype(str)
    )


    psi = calcular_psi_categorico(
        referencia,
        actual
    )


    js = calcular_js_categorico(
        referencia,
        actual
    )


    chi2, chi2_p_value = (
        calcular_chi_cuadrado(
            referencia,
            actual
        )
    )


    drift_psi = (
        not pd.isna(psi)
        and psi >= 0.25
    )


    drift_js = (
        not pd.isna(js)
        and js >= 0.10
    )


    drift_chi2 = (
        not pd.isna(chi2_p_value)
        and chi2_p_value < 0.05
    )


    drift_detectado = (
        drift_psi
        or
        drift_js
        or
        drift_chi2
    )


    if (
        not pd.isna(psi)
        and psi >= 0.25
    ):

        nivel_drift = "Alto"

    elif (
        not pd.isna(psi)
        and psi >= 0.10
    ):

        nivel_drift = "Moderado"

    elif drift_detectado:

        nivel_drift = "Detectado"

    else:

        nivel_drift = "Estable"


    resultados_drift.append(
        {
            "variable": columna,
            "tipo_variable": "Categorica",
            "ks_statistic": np.nan,
            "ks_p_value": np.nan,
            "psi": psi,
            "jensen_shannon": js,
            "chi2": chi2,
            "chi2_p_value": chi2_p_value,
            "drift_ks": False,
            "drift_psi": drift_psi,
            "drift_js": drift_js,
            "drift_chi2": drift_chi2,
            "drift_detectado": drift_detectado,
            "nivel_drift": nivel_drift,
            "registros_referencia": len(referencia),
            "registros_actuales": len(actual)
        }
    )


# ============================================================
# RESULTADOS DE DATA DRIFT
# ============================================================

resultados_drift_df = pd.DataFrame(
    resultados_drift
)


if len(resultados_drift_df) > 0:

    resultados_drift_df = (
        resultados_drift_df
        .sort_values(
            by=[
                "drift_detectado",
                "psi"
            ],
            ascending=[
                False,
                False
            ]
        )
    )


resultados_drift_df.to_csv(
    RUTA_DRIFT,
    index=False
)


# ============================================================
# VARIABLES CON DRIFT
# ============================================================

variables_con_drift = (
    resultados_drift_df[
        resultados_drift_df[
            "drift_detectado"
        ]
    ].copy()
)


variables_con_drift.to_csv(
    RUTA_VARIABLES_DRIFT,
    index=False
)


# ============================================================
# ALERTAS DE DATA DRIFT
# ============================================================

alertas_drift = []


if len(resultados_drift_df) == 0:

    alertas_drift.append(
        "No fue posible calcular métricas de data drift."
    )

else:

    cantidad_variables_drift = int(
        resultados_drift_df[
            "drift_detectado"
        ].sum()
    )


    cantidad_variables_monitorizadas = (
        len(resultados_drift_df)
    )


    if cantidad_variables_drift > 0:

        alertas_drift.append(
            f"Se detectó data drift en "
            f"{cantidad_variables_drift} de "
            f"{cantidad_variables_monitorizadas} "
            f"variables monitorizadas."
        )

    else:

        alertas_drift.append(
            "No se detectó data drift significativo "
            "en las variables monitorizadas."
        )


    variables_drift_alto = (
        resultados_drift_df[
            resultados_drift_df[
                "nivel_drift"
            ] == "Alto"
        ]
    )


    if len(variables_drift_alto) > 0:

        nombres = ", ".join(
            variables_drift_alto[
                "variable"
            ].astype(str)
            .tolist()
        )

        alertas_drift.append(
            "Variables con nivel de drift alto: "
            f"{nombres}."
        )


    variables_ks = (
        resultados_drift_df[
            resultados_drift_df[
                "drift_ks"
            ]
        ]
    )


    if len(variables_ks) > 0:

        nombres = ", ".join(
            variables_ks[
                "variable"
            ].astype(str)
            .tolist()
        )

        alertas_drift.append(
            "Variables con KS significativo: "
            f"{nombres}."
        )


    variables_chi = (
        resultados_drift_df[
            resultados_drift_df[
                "drift_chi2"
            ]
        ]
    )


    if len(variables_chi) > 0:

        nombres = ", ".join(
            variables_chi[
                "variable"
            ].astype(str)
            .tolist()
        )

        alertas_drift.append(
            "Variables categóricas con Chi-cuadrado "
            f"significativo: {nombres}."
        )


with open(
    RUTA_DRIFT_ALERTAS,
    "w",
    encoding="utf-8"
) as archivo_drift:

    archivo_drift.write(
        "Alertas de Data Drift\n"
    )

    archivo_drift.write(
        "=====================\n\n"
    )

    for numero, alerta in enumerate(
        alertas_drift,
        start=1
    ):

        archivo_drift.write(
            f"{numero}. {alerta}\n"
        )


# ============================================================
# GRÁFICOS DE DATA DRIFT
# ============================================================

for columna in columnas_numericas_drift:

    referencia = pd.to_numeric(
        df_referencia[columna],
        errors="coerce"
    ).dropna()

    actual = pd.to_numeric(
        df_validas[columna],
        errors="coerce"
    ).dropna()


    if len(referencia) == 0 or len(actual) == 0:

        continue


    plt.figure(
        figsize=(8, 5)
    )


    plt.hist(
        referencia,
        bins=20,
        alpha=0.5,
        density=True,
        label="Referencia"
    )


    plt.hist(
        actual,
        bins=20,
        alpha=0.5,
        density=True,
        label="Actual"
    )


    plt.title(
        f"Data Drift - {columna}"
    )

    plt.xlabel(
        columna
    )

    plt.ylabel(
        "Densidad"
    )

    plt.legend()

    plt.tight_layout()


    ruta_grafico = os.path.join(
        CARPETA_DRIFT,
        f"drift_{columna}.png"
    )


    plt.savefig(
        ruta_grafico,
        dpi=300
    )

    plt.close()


# ============================================================
# ANÁLISIS TEMPORAL
# ============================================================

if "fecha_prediccion_api" in df_validas.columns:

    fechas_api = pd.to_datetime(
        df_validas[
            "fecha_prediccion_api"
        ],
        errors="coerce"
    )

    df_temporal = df_validas.copy()

    df_temporal[
        "fecha_prediccion_api"
    ] = fechas_api

    df_temporal = df_temporal.dropna(
        subset=[
            "fecha_prediccion_api"
        ]
    )


    if len(df_temporal) > 0:

        df_temporal[
            "periodo"
        ] = (
            df_temporal[
                "fecha_prediccion_api"
            ]
            .dt
            .to_period("D")
            .astype(str)
        )


        resumen_temporal = (
            df_temporal
            .groupby("periodo")
            .agg(
                total_predicciones=(
                    "prediccion_modelo",
                    "count"
                ),

                promedio_probabilidad_incumplimiento=(
                    "probabilidad_incumplimiento",
                    "mean"
                ),

                porcentaje_incumplimiento=(
                    "prediccion_modelo",
                    lambda x:
                    (x == 0).mean() * 100
                ),

                porcentaje_pago_atiempo=(
                    "prediccion_modelo",
                    lambda x:
                    (x == 1).mean() * 100
                )
            )
            .reset_index()
        )


        resumen_temporal.to_csv(
            RUTA_RESUMEN_TEMPORAL,
            index=False
        )


        plt.figure(
            figsize=(10, 5)
        )


        plt.plot(
            resumen_temporal[
                "periodo"
            ],
            resumen_temporal[
                "porcentaje_incumplimiento"
            ],
            marker="o",
            label="Incumplimiento"
        )


        plt.plot(
            resumen_temporal[
                "periodo"
            ],
            resumen_temporal[
                "porcentaje_pago_atiempo"
            ],
            marker="o",
            label="Pago a tiempo"
        )


        plt.title(
            "Evolución temporal de las predicciones"
        )

        plt.xlabel(
            "Periodo"
        )

        plt.ylabel(
            "Porcentaje"
        )

        plt.xticks(
            rotation=45
        )

        plt.legend()

        plt.tight_layout()


        ruta_temporal = os.path.join(
            CARPETA_RESULTADOS,
            "analisis_temporal_predicciones.png"
        )


        plt.savefig(
            ruta_temporal,
            dpi=300
        )

        plt.close()


# ============================================================
# ALERTAS BÁSICAS
# ============================================================

alertas = []


if total_validos < 30:

    alertas.append(
        "Muestra pequeña: se recomienda aumentar "
        "la cantidad de predicciones."
    )


if porcentaje_incumplimiento >= 80:

    alertas.append(
        "Alta proporción de predicciones clasificadas "
        "como posible incumplimiento."
    )


if promedio_probabilidad_incumplimiento >= 0.70:

    alertas.append(
        "La probabilidad promedio de incumplimiento "
        "se encuentra en nivel alto."
    )


if porcentaje_riesgo_alto >= 50:

    alertas.append(
        "Una proporción elevada de las predicciones "
        "presenta nivel de riesgo alto."
    )


if total_invalidos > 0:

    alertas.append(
        f"Se detectaron {total_invalidos} "
        "registros inválidos o incompletos."
    )


if len(variables_con_drift) > 0:

    alertas.append(
        f"Se detectó data drift en "
        f"{len(variables_con_drift)} variables."
    )


if len(alertas) == 0:

    alertas.append(
        "No se detectaron alertas básicas."
    )


# ============================================================
# TABLA GENERAL DE MÉTRICAS
# ============================================================

cantidad_variables_monitorizadas = (
    len(resultados_drift_df)
)


cantidad_variables_con_drift = (
    len(variables_con_drift)
)


porcentaje_variables_con_drift = 0


if cantidad_variables_monitorizadas > 0:

    porcentaje_variables_con_drift = (
        cantidad_variables_con_drift /
        cantidad_variables_monitorizadas
    ) * 100


metricas = pd.DataFrame(
    {
        "metrica": [

            "total_registros_archivo",

            "total_predicciones_validas",

            "total_predicciones_invalidas",

            "predicciones_incumplimiento",

            "predicciones_pago_atiempo",

            "porcentaje_incumplimiento",

            "porcentaje_pago_atiempo",

            "promedio_probabilidad_incumplimiento",

            "mediana_probabilidad_incumplimiento",

            "desviacion_probabilidad_incumplimiento",

            "minima_probabilidad_incumplimiento",

            "maxima_probabilidad_incumplimiento",

            "promedio_probabilidad_pago_atiempo",

            "promedio_tiempo_prediccion_ms",

            "mediana_tiempo_prediccion_ms",

            "minimo_tiempo_prediccion_ms",

            "maximo_tiempo_prediccion_ms",

            "cantidad_riesgo_alto",

            "porcentaje_riesgo_alto",

            "cantidad_variables_monitorizadas_drift",

            "cantidad_variables_con_drift",

            "porcentaje_variables_con_drift",

            "cantidad_alertas"
        ],

        "valor": [

            total_registros,

            total_validos,

            total_invalidos,

            cantidad_incumplimiento,

            cantidad_pago_atiempo,

            porcentaje_incumplimiento,

            porcentaje_pago_atiempo,

            promedio_probabilidad_incumplimiento,

            mediana_probabilidad_incumplimiento,

            desviacion_probabilidad_incumplimiento,

            minima_probabilidad_incumplimiento,

            maxima_probabilidad_incumplimiento,

            promedio_probabilidad_pago_atiempo,

            promedio_tiempo_ms,

            mediana_tiempo_ms,

            minimo_tiempo_ms,

            maximo_tiempo_ms,

            cantidad_riesgo_alto,

            porcentaje_riesgo_alto,

            cantidad_variables_monitorizadas,

            cantidad_variables_con_drift,

            porcentaje_variables_con_drift,

            len(alertas)
        ]
    }
)


metricas.to_csv(
    RUTA_METRICAS,
    index=False
)


# ============================================================
# GUARDAR ALERTAS GENERALES
# ============================================================

with open(
    RUTA_ALERTAS,
    "w",
    encoding="utf-8"
) as archivo_alertas:

    archivo_alertas.write(
        "Alertas del model monitoring\n"
    )

    archivo_alertas.write(
        "============================\n\n"
    )

    for numero, alerta in enumerate(
        alertas,
        start=1
    ):

        archivo_alertas.write(
            f"{numero}. {alerta}\n"
        )


# ============================================================
# GRÁFICO DE DISTRIBUCIÓN DE PREDICCIONES
# ============================================================

plt.figure(
    figsize=(8, 5)
)


plt.bar(
    distribucion_predicciones[
        "resultado_prediccion"
    ],
    distribucion_predicciones[
        "cantidad"
    ]
)


plt.title(
    "Distribución de predicciones del modelo"
)

plt.xlabel(
    "Resultado de la predicción"
)

plt.ylabel(
    "Cantidad de registros"
)

plt.xticks(
    rotation=10
)

plt.tight_layout()


plt.savefig(
    RUTA_GRAFICO_PREDICCIONES,
    dpi=300
)

plt.close()


# ============================================================
# GRÁFICO DE PROBABILIDADES
# ============================================================

plt.figure(
    figsize=(8, 5)
)


plt.hist(
    df_validas[
        "probabilidad_incumplimiento"
    ],
    bins=10
)


plt.title(
    "Distribución de probabilidad de incumplimiento"
)

plt.xlabel(
    "Probabilidad de incumplimiento"
)

plt.ylabel(
    "Cantidad de registros"
)

plt.tight_layout()


plt.savefig(
    RUTA_GRAFICO_PROBABILIDADES,
    dpi=300
)

plt.close()


# ============================================================
# GRÁFICO DE NIVELES DE RIESGO
# ============================================================

plt.figure(
    figsize=(8, 5)
)


plt.bar(
    resumen_riesgo[
        "nivel_riesgo"
    ],
    resumen_riesgo[
        "cantidad"
    ]
)


plt.title(
    "Distribución de niveles de riesgo"
)

plt.xlabel(
    "Nivel de riesgo"
)

plt.ylabel(
    "Cantidad de registros"
)

plt.tight_layout()


plt.savefig(
    RUTA_GRAFICO_RIESGO,
    dpi=300
)

plt.close()


# ============================================================
# MOSTRAR RESULTADOS
# ============================================================

print(
    "Resultados del monitoreo"
)

print()


print(
    f"Total de registros en el archivo: "
    f"{total_registros}"
)


print(
    f"Predicciones válidas: "
    f"{total_validos}"
)


print(
    f"Predicciones inválidas: "
    f"{total_invalidos}"
)


print()


print(
    f"Predicciones de posible incumplimiento: "
    f"{cantidad_incumplimiento} "
    f"({porcentaje_incumplimiento:.2f}%)"
)


print(
    f"Predicciones de pago a tiempo: "
    f"{cantidad_pago_atiempo} "
    f"({porcentaje_pago_atiempo:.2f}%)"
)


print()


print(
    f"Promedio probabilidad de incumplimiento: "
    f"{promedio_probabilidad_incumplimiento:.4f}"
)


print(
    f"Mediana probabilidad de incumplimiento: "
    f"{mediana_probabilidad_incumplimiento:.4f}"
)


print(
    f"Probabilidad mínima de incumplimiento: "
    f"{minima_probabilidad_incumplimiento:.4f}"
)


print(
    f"Probabilidad máxima de incumplimiento: "
    f"{maxima_probabilidad_incumplimiento:.4f}"
)


print()


print(
    "Distribución de niveles de riesgo:"
)


for _, fila in resumen_riesgo.iterrows():

    print(
        f"- {fila['nivel_riesgo']}: "
        f"{int(fila['cantidad'])} "
        f"({fila['porcentaje']:.2f}%)"
    )


print()


print(
    f"Predicciones con riesgo alto: "
    f"{cantidad_riesgo_alto} "
    f"({porcentaje_riesgo_alto:.2f}%)"
)


print()


if not pd.isna(
    promedio_tiempo_ms
):

    print(
        f"Tiempo promedio de predicción: "
        f"{promedio_tiempo_ms:.4f} ms"
    )

    print(
        f"Mediana del tiempo de predicción: "
        f"{mediana_tiempo_ms:.4f} ms"
    )

    print(
        f"Tiempo mínimo de predicción: "
        f"{minimo_tiempo_ms:.4f} ms"
    )

    print(
        f"Tiempo máximo de predicción: "
        f"{maximo_tiempo_ms:.4f} ms"
    )

else:

    print(
        "Tiempo de predicción: "
        "No disponible en el archivo."
    )


print()


print(
    "Resultados de Data Drift:"
)

print(
    f"- Variables monitorizadas: "
    f"{cantidad_variables_monitorizadas}"
)

print(
    f"- Variables con drift: "
    f"{cantidad_variables_con_drift}"
)

print(
    f"- Porcentaje con drift: "
    f"{porcentaje_variables_con_drift:.2f}%"
)


print()


print(
    "Alertas:"
)


for alerta in alertas:

    print(
        f"- {alerta}"
    )


print()


print(
    "Archivos de Data Drift generados:"
)

print(
    RUTA_DRIFT
)

print(
    RUTA_VARIABLES_DRIFT
)

print(
    RUTA_DRIFT_ALERTAS
)

print()


print(
    "Archivos generados en:"
)

print(
    CARPETA_RESULTADOS
)

print()


print(
    "Model monitoring finalizado correctamente."
)