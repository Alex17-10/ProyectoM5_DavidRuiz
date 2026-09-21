import os
import json
import joblib
import numpy as np
import pandas as pd

from datetime import datetime
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# Definir la carpeta principal del proyecto
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# Definir las rutas de los archivos
RUTA_MODELO = os.path.join(
    BASE_DIR,
    "modelo_final.joblib"
)

RUTA_PREPROCESADOR = os.path.join(
    BASE_DIR,
    "resultados_feature_engineering",
    "pipeline_preprocesamiento.joblib"
)

RUTA_DATOS = os.path.join(
    BASE_DIR,
    "Base_de_datos_preparada.csv"
)

CARPETA_SALIDA = os.path.join(
    BASE_DIR,
    "resultados_model_deploy"
)

os.makedirs(
    CARPETA_SALIDA,
    exist_ok=True
)


# Verificar que existan los archivos necesarios
archivos_necesarios = {
    "Modelo final": RUTA_MODELO,
    "Preprocesador": RUTA_PREPROCESADOR,
    "Base de datos": RUTA_DATOS
}

for nombre_archivo, ruta_archivo in archivos_necesarios.items():
    if not os.path.exists(ruta_archivo):
        raise FileNotFoundError(
            f"No se encontró {nombre_archivo} en: {ruta_archivo}"
        )


# Cargar el modelo y el preprocesador
modelo = joblib.load(
    RUTA_MODELO
)

preprocesador = joblib.load(
    RUTA_PREPROCESADOR
)

print("\n" + "=" * 60)
print("MODEL DEPLOY")
print("=" * 60)

print("\nArchivos cargados correctamente")
print(
    f"{'Modelo:':<25} "
    f"{os.path.basename(RUTA_MODELO)}"
)

print(
    f"{'Preprocesador:':<25} "
    f"{os.path.basename(RUTA_PREPROCESADOR)}"
)


# Cargar los datos
df = pd.read_csv(
    RUTA_DATOS
)

print("\nDatos cargados correctamente")
print(
    f"{'Cantidad de registros:':<25} "
    f"{df.shape[0]}"
)

print(
    f"{'Cantidad de columnas:':<25} "
    f"{df.shape[1]}"
)


# Mostrar las columnas originales
print("\nColumnas originales")
print("-" * 60)

for columna in df.columns:
    print(f"- {columna}")


# Crear una copia para realizar las transformaciones
df_procesado = df.copy()


# Convertir la fecha y crear variables derivadas
if "fecha_prestamo" in df_procesado.columns:

    df_procesado["fecha_prestamo"] = pd.to_datetime(
        df_procesado["fecha_prestamo"],
        errors="coerce"
    )

    df_procesado["anio_prestamo"] = (
        df_procesado["fecha_prestamo"].dt.year
    )

    df_procesado["mes_prestamo"] = (
        df_procesado["fecha_prestamo"].dt.month
    )

    df_procesado["dia_prestamo"] = (
        df_procesado["fecha_prestamo"].dt.day
    )

    df_procesado["dia_semana_prestamo"] = (
        df_procesado["fecha_prestamo"].dt.dayofweek
    )

    df_procesado["semana_anio_prestamo"] = (
        df_procesado["fecha_prestamo"]
        .dt.isocalendar()
        .week
        .astype("float")
    )

    df_procesado["trimestre_prestamo"] = (
        df_procesado["fecha_prestamo"].dt.quarter
    )

    df_procesado["hora_prestamo"] = (
        df_procesado["fecha_prestamo"].dt.hour
        + df_procesado["fecha_prestamo"].dt.minute / 60
    )

    df_procesado["es_fin_semana"] = (
        df_procesado["fecha_prestamo"].dt.dayofweek >= 5
    ).astype(int)


# Crear la relación entre cuota e ingreso
if {
    "cuota_mensual",
    "ingreso_mensual"
}.issubset(df_procesado.columns):

    df_procesado["relacion_cuota_salario"] = (
        df_procesado["cuota_mensual"]
        / df_procesado["ingreso_mensual"].replace(0, np.nan)
    )


# Crear la relación entre deuda e ingreso
if {
    "deuda_total",
    "ingreso_mensual"
}.issubset(df_procesado.columns):

    df_procesado["relacion_deuda_salario"] = (
        df_procesado["deuda_total"]
        / df_procesado["ingreso_mensual"].replace(0, np.nan)
    )


# Estimar el saldo pendiente
if {
    "saldo_principal",
    "cuotas_pagadas",
    "cuotas_totales"
}.issubset(df_procesado.columns):

    df_procesado["saldo_pendiente_estimado"] = (
        df_procesado["saldo_principal"]
        * (
            1
            - (
                df_procesado["cuotas_pagadas"]
                / df_procesado["cuotas_totales"].replace(
                    0,
                    np.nan
                )
            )
        )
    )


# Separar la variable objetivo
TARGET = "Pago_atiempo"

if TARGET in df_procesado.columns:

    y_real = df_procesado[TARGET].copy()

    X = df_procesado.drop(
        columns=[TARGET]
    )

else:

    y_real = None

    X = df_procesado.copy()


# Eliminar las variables que no se utilizaron en el entrenamiento
if "puntaje" in X.columns:

    X = X.drop(
        columns=["puntaje"]
    )

if "fecha_prestamo" in X.columns:

    X = X.drop(
        columns=["fecha_prestamo"]
    )


# Mostrar la cantidad de características antes del preprocesamiento
print("\nPreparación de las características")
print("-" * 60)

print(
    f"{'Características de entrada:':<35} "
    f"{X.shape[1]}"
)


# Aplicar el mismo preprocesamiento utilizado durante el entrenamiento
X_transformado = preprocesador.transform(
    X
)


# Recuperar los nombres utilizados por el modelo durante el entrenamiento
if hasattr(modelo, "feature_names_in_"):

    nombres_modelo = modelo.feature_names_in_

    X_transformado = pd.DataFrame(
        X_transformado,
        columns=nombres_modelo,
        index=X.index
    )

else:

    X_transformado = np.asarray(
        X_transformado
    )

print(
    f"{'Características transformadas:':<35} "
    f"{X_transformado.shape[1]}"
)


# Verificar que la cantidad de características coincida
if hasattr(modelo, "n_features_in_"):

    cantidad_esperada = modelo.n_features_in_
    cantidad_recibida = X_transformado.shape[1]

    if cantidad_esperada != cantidad_recibida:

        raise ValueError(
            "La cantidad de características no coincide. "
            f"El modelo espera {cantidad_esperada}, "
            f"pero recibió {cantidad_recibida}."
        )

    print(
        f"{'Características esperadas por el modelo:':<35} "
        f"{cantidad_esperada}"
    )


# Generar las predicciones
predicciones = modelo.predict(
    X_transformado
)


# Verificar si el modelo permite obtener probabilidades
tiene_probabilidades = hasattr(
    modelo,
    "predict_proba"
)

if tiene_probabilidades:

    probabilidades = modelo.predict_proba(
        X_transformado
    )

    # Identificar las posiciones de las clases
    clases_modelo = list(
        modelo.classes_
    )

    if 0 in clases_modelo:

        posicion_clase_0 = clases_modelo.index(0)

        probabilidad_incumplimiento = (
            probabilidades[:, posicion_clase_0]
        )

    else:

        probabilidad_incumplimiento = np.nan

    if 1 in clases_modelo:

        posicion_clase_1 = clases_modelo.index(1)

        probabilidad_pago_atiempo = (
            probabilidades[:, posicion_clase_1]
        )

    else:

        probabilidad_pago_atiempo = np.nan


# Crear el DataFrame con los resultados
df_resultados = df.copy()

df_resultados["prediccion_modelo"] = (
    predicciones
)

if tiene_probabilidades:

    df_resultados["probabilidad_incumplimiento"] = (
        probabilidad_incumplimiento
    )

    df_resultados["probabilidad_pago_atiempo"] = (
        probabilidad_pago_atiempo
    )

df_resultados["resultado_prediccion"] = np.where(
    df_resultados["prediccion_modelo"] == 1,
    "Pago a tiempo",
    "Posible incumplimiento"
)


# Calcular la distribución de las predicciones
cantidad_total = len(
    df_resultados
)

cantidad_predicha_0 = int(
    (
        df_resultados["prediccion_modelo"] == 0
    ).sum()
)

cantidad_predicha_1 = int(
    (
        df_resultados["prediccion_modelo"] == 1
    ).sum()
)

porcentaje_predicha_0 = (
    cantidad_predicha_0
    / cantidad_total
    * 100
)

porcentaje_predicha_1 = (
    cantidad_predicha_1
    / cantidad_total
    * 100
)


# Mostrar la distribución real de la variable objetivo
print("\nDistribución real de los datos")
print("-" * 60)

if y_real is not None:

    cantidad_real_0 = int(
        (y_real == 0).sum()
    )

    cantidad_real_1 = int(
        (y_real == 1).sum()
    )

    porcentaje_real_0 = (
        cantidad_real_0
        / len(y_real)
        * 100
    )

    porcentaje_real_1 = (
        cantidad_real_1
        / len(y_real)
        * 100
    )

    print(
        f"{'Clase 0 - Incumplimiento:':<35} "
        f"{cantidad_real_0:>6} "
        f"({porcentaje_real_0:>6.2f}%)"
    )

    print(
        f"{'Clase 1 - Pago a tiempo:':<35} "
        f"{cantidad_real_1:>6} "
        f"({porcentaje_real_1:>6.2f}%)"
    )

else:

    cantidad_real_0 = None
    cantidad_real_1 = None
    porcentaje_real_0 = None
    porcentaje_real_1 = None

    print(
        "La base no contiene la variable objetivo."
    )


# Mostrar la distribución de las predicciones
print("\nDistribución de las predicciones")
print("-" * 60)

print(
    f"{'Clase 0 - Posible incumplimiento:':<35} "
    f"{cantidad_predicha_0:>6} "
    f"({porcentaje_predicha_0:>6.2f}%)"
)

print(
    f"{'Clase 1 - Pago a tiempo:':<35} "
    f"{cantidad_predicha_1:>6} "
    f"({porcentaje_predicha_1:>6.2f}%)"
)


# Evaluar las predicciones si se dispone de los valores reales
metricas = {}

if y_real is not None:

    accuracy = accuracy_score(
        y_real,
        predicciones
    )

    balanced_accuracy = balanced_accuracy_score(
        y_real,
        predicciones
    )

    precision_clase_0 = precision_score(
        y_real,
        predicciones,
        pos_label=0,
        zero_division=0
    )

    recall_clase_0 = recall_score(
        y_real,
        predicciones,
        pos_label=0,
        zero_division=0
    )

    f1_clase_0 = f1_score(
        y_real,
        predicciones,
        pos_label=0,
        zero_division=0
    )

    precision_clase_1 = precision_score(
        y_real,
        predicciones,
        pos_label=1,
        zero_division=0
    )

    recall_clase_1 = recall_score(
        y_real,
        predicciones,
        pos_label=1,
        zero_division=0
    )

    f1_clase_1 = f1_score(
        y_real,
        predicciones,
        pos_label=1,
        zero_division=0
    )

    metricas = {
        "accuracy": round(float(accuracy), 4),
        "balanced_accuracy": round(
            float(balanced_accuracy),
            4
        ),
        "precision_clase_0": round(
            float(precision_clase_0),
            4
        ),
        "recall_clase_0": round(
            float(recall_clase_0),
            4
        ),
        "f1_clase_0": round(
            float(f1_clase_0),
            4
        ),
        "precision_clase_1": round(
            float(precision_clase_1),
            4
        ),
        "recall_clase_1": round(
            float(recall_clase_1),
            4
        ),
        "f1_clase_1": round(
            float(f1_clase_1),
            4
        )
    }

    print("\nMétricas sobre la base utilizada")
    print("-" * 60)

    print(
        f"{'Accuracy:':<35} "
        f"{accuracy:.4f}"
    )

    print(
        f"{'Balanced Accuracy:':<35} "
        f"{balanced_accuracy:.4f}"
    )

    print(
        f"{'Precision clase 0:':<35} "
        f"{precision_clase_0:.4f}"
    )

    print(
        f"{'Recall clase 0:':<35} "
        f"{recall_clase_0:.4f}"
    )

    print(
        f"{'F1 clase 0:':<35} "
        f"{f1_clase_0:.4f}"
    )

    print(
        f"{'Precision clase 1:':<35} "
        f"{precision_clase_1:.4f}"
    )

    print(
        f"{'Recall clase 1:':<35} "
        f"{recall_clase_1:.4f}"
    )

    print(
        f"{'F1 clase 1:':<35} "
        f"{f1_clase_1:.4f}"
    )

    # Matriz de confusión
    matriz = confusion_matrix(
        y_real,
        predicciones,
        labels=[0, 1]
    )

    print("\nMatriz de confusión")
    print("-" * 60)

    matriz_df = pd.DataFrame(
        matriz,
        index=[
            "Real 0",
            "Real 1"
        ],
        columns=[
            "Predicho 0",
            "Predicho 1"
        ]
    )

    print(
        matriz_df.to_string()
    )

else:

    print("\nNo se calcularon métricas")
    print(
        "La base no contiene la variable Pago_atiempo."
    )


# Mostrar una vista previa de los resultados
print("\nVista previa de las predicciones")
print("-" * 60)

columnas_vista = [
    "prediccion_modelo",
    "resultado_prediccion"
]

if tiene_probabilidades:

    columnas_vista.extend([
        "probabilidad_incumplimiento",
        "probabilidad_pago_atiempo"
    ])

vista_previa = df_resultados[
    columnas_vista
].head(10).copy()

if tiene_probabilidades:

    vista_previa[
        "probabilidad_incumplimiento"
    ] = vista_previa[
        "probabilidad_incumplimiento"
    ].round(4)

    vista_previa[
        "probabilidad_pago_atiempo"
    ] = vista_previa[
        "probabilidad_pago_atiempo"
    ].round(4)

print(
    vista_previa.to_string(
        index=True
    )
)


# Guardar las predicciones completas
RUTA_RESULTADOS = os.path.join(
    CARPETA_SALIDA,
    "predicciones_modelo_final.csv"
)

df_resultados.to_csv(
    RUTA_RESULTADOS,
    index=False,
    encoding="utf-8-sig"
)


# Crear el resumen general
resumen = {
    "fecha_ejecucion": datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    ),
    "modelo_utilizado": os.path.basename(
        RUTA_MODELO
    ),
    "preprocesador_utilizado": os.path.basename(
        RUTA_PREPROCESADOR
    ),
    "archivo_datos": os.path.basename(
        RUTA_DATOS
    ),
    "cantidad_registros": int(
        cantidad_total
    ),
    "cantidad_columnas_originales": int(
        df.shape[1]
    ),
    "cantidad_caracteristicas_entrada": int(
        X.shape[1]
    ),
    "cantidad_caracteristicas_transformadas": int(
        X_transformado.shape[1]
    ),
    "predicciones_clase_0": int(
        cantidad_predicha_0
    ),
    "porcentaje_predicciones_clase_0": round(
        float(porcentaje_predicha_0),
        2
    ),
    "predicciones_clase_1": int(
        cantidad_predicha_1
    ),
    "porcentaje_predicciones_clase_1": round(
        float(porcentaje_predicha_1),
        2
    ),
    "metricas": metricas
}


# Guardar el resumen en formato JSON
RUTA_RESUMEN_JSON = os.path.join(
    CARPETA_SALIDA,
    "resumen_model_deploy.json"
)

with open(
    RUTA_RESUMEN_JSON,
    "w",
    encoding="utf-8"
) as archivo:

    json.dump(
        resumen,
        archivo,
        ensure_ascii=False,
        indent=4
    )


# Crear un resumen tabulado
filas_resumen = [
    [
        "Registros procesados",
        cantidad_total
    ],
    [
        "Características de entrada",
        X.shape[1]
    ],
    [
        "Características transformadas",
        X_transformado.shape[1]
    ],
    [
        "Predicciones clase 0",
        cantidad_predicha_0
    ],
    [
        "Porcentaje predicciones clase 0",
        f"{porcentaje_predicha_0:.2f}%"
    ],
    [
        "Predicciones clase 1",
        cantidad_predicha_1
    ],
    [
        "Porcentaje predicciones clase 1",
        f"{porcentaje_predicha_1:.2f}%"
    ]
]

if metricas:

    filas_resumen.extend([
        [
            "Accuracy",
            metricas["accuracy"]
        ],
        [
            "Balanced Accuracy",
            metricas["balanced_accuracy"]
        ],
        [
            "Recall clase 0",
            metricas["recall_clase_0"]
        ],
        [
            "F1 clase 0",
            metricas["f1_clase_0"]
        ],
        [
            "Recall clase 1",
            metricas["recall_clase_1"]
        ],
        [
            "F1 clase 1",
            metricas["f1_clase_1"]
        ]
    ])

df_resumen = pd.DataFrame(
    filas_resumen,
    columns=[
        "Indicador",
        "Valor"
    ]
)

RUTA_RESUMEN_CSV = os.path.join(
    CARPETA_SALIDA,
    "resumen_model_deploy.csv"
)

df_resumen.to_csv(
    RUTA_RESUMEN_CSV,
    index=False,
    encoding="utf-8-sig"
)


# Crear un reporte de texto
RUTA_REPORTE = os.path.join(
    CARPETA_SALIDA,
    "reporte_model_deploy.txt"
)

with open(
    RUTA_REPORTE,
    "w",
    encoding="utf-8"
) as archivo:

    archivo.write("REPORTE MODEL DEPLOY\n")
    archivo.write("=" * 60 + "\n\n")

    archivo.write(
        f"Fecha de ejecución: "
        f"{resumen['fecha_ejecucion']}\n"
    )

    archivo.write(
        f"Modelo utilizado: "
        f"{resumen['modelo_utilizado']}\n"
    )

    archivo.write(
        f"Preprocesador utilizado: "
        f"{resumen['preprocesador_utilizado']}\n"
    )

    archivo.write(
        f"Archivo de datos: "
        f"{resumen['archivo_datos']}\n\n"
    )

    archivo.write("DATOS PROCESADOS\n")
    archivo.write("-" * 60 + "\n")

    archivo.write(
        f"Registros procesados: "
        f"{cantidad_total}\n"
    )

    archivo.write(
        f"Características de entrada: "
        f"{X.shape[1]}\n"
    )

    archivo.write(
        f"Características transformadas: "
        f"{X_transformado.shape[1]}\n\n"
    )

    archivo.write("DISTRIBUCIÓN DE LAS PREDICCIONES\n")
    archivo.write("-" * 60 + "\n")

    archivo.write(
        f"Clase 0 - Posible incumplimiento: "
        f"{cantidad_predicha_0} "
        f"({porcentaje_predicha_0:.2f}%)\n"
    )

    archivo.write(
        f"Clase 1 - Pago a tiempo: "
        f"{cantidad_predicha_1} "
        f"({porcentaje_predicha_1:.2f}%)\n\n"
    )

    if metricas:

        archivo.write("MÉTRICAS\n")
        archivo.write("-" * 60 + "\n")

        for nombre_metrica, valor in metricas.items():

            archivo.write(
                f"{nombre_metrica}: {valor}\n"
            )

        archivo.write("\nMATRIZ DE CONFUSIÓN\n")
        archivo.write("-" * 60 + "\n")
        archivo.write(
            matriz_df.to_string()
        )
        archivo.write("\n")


# Mostrar las rutas de los archivos generados
print("\nArchivos generados")
print("-" * 60)

print(
    f"{'Predicciones:':<25} "
    f"{RUTA_RESULTADOS}"
)

print(
    f"{'Resumen JSON:':<25} "
    f"{RUTA_RESUMEN_JSON}"
)

print(
    f"{'Resumen CSV:':<25} "
    f"{RUTA_RESUMEN_CSV}"
)

print(
    f"{'Reporte de texto:':<25} "
    f"{RUTA_REPORTE}"
)

print("\nProceso model_deploy finalizado correctamente")