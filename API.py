import os
import time
import uuid
from datetime import datetime
from typing import List, Optional

import joblib
import pandas as pd
import uvicorn

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# Rutas del proyecto

CARPETA_PROYECTO = os.path.dirname(os.path.abspath(__file__))

RUTA_MODELO = os.path.join(
    CARPETA_PROYECTO,
    "modelo_final.joblib"
)

RUTA_PIPELINE = os.path.join(
    CARPETA_PROYECTO,
    "resultados_feature_engineering",
    "pipeline_preprocesamiento.joblib"
)

ARCHIVO_PREDICCIONES = os.path.join(
    CARPETA_PROYECTO,
    "predicciones_api_monitoring.csv"
)


# Carga del modelo y pipeline

modelo = None
pipeline_preprocesamiento = None

try:
    modelo = joblib.load(RUTA_MODELO)
    pipeline_preprocesamiento = joblib.load(RUTA_PIPELINE)

    print("Modelo y pipeline cargados correctamente.")

except Exception as error:
    print(f"Error al cargar el modelo o el pipeline: {error}")
    print(f"Ruta del modelo: {RUTA_MODELO}")
    print(f"Ruta del pipeline: {RUTA_PIPELINE}")


# Aplicación FastAPI

app = FastAPI(
    title="API de Predicción de Pago de Créditos",
    description=(
        "API para predecir si un crédito será pagado a tiempo "
        "utilizando un modelo de machine learning."
    ),
    version="1.2.0"
)


# Esquemas de entrada

class RegistroCredito(BaseModel):
    tipo_credito: str
    fecha_prestamo: str
    capital_prestado: float
    plazo_meses: int
    edad_cliente: int
    tipo_laboral: str
    salario_cliente: float
    total_otros_prestamos: float
    cuota_pactada: float
    puntaje_datacredito: float
    cant_creditosvigentes: int
    huella_consulta: int
    saldo_mora: float
    saldo_total: float
    saldo_principal: float
    saldo_mora_codeudor: float
    creditos_sectorFinanciero: int
    creditos_sectorCooperativo: int
    creditos_sectorReal: int
    promedio_ingresos_datacredito: float
    tendencia_ingresos: Optional[str] = None


class SolicitudBatch(BaseModel):
    registros: List[RegistroCredito] = Field(
        ...,
        min_length=1,
        description="Lista de registros de créditos para predecir."
    )


# Esquemas de salida

class ResultadoPrediccion(BaseModel):
    id_prediccion: str
    prediccion_modelo: int
    resultado_prediccion: str
    probabilidad_incumplimiento: float
    probabilidad_pago_atiempo: float
    nivel_riesgo: str
    tiempo_prediccion_ms: float


class RespuestaBatch(BaseModel):
    cantidad_registros: int
    predicciones: List[ResultadoPrediccion]


# Columnas del archivo CSV

COLUMNAS_CSV = [
    "tipo_credito",
    "fecha_prestamo",
    "capital_prestado",
    "plazo_meses",
    "edad_cliente",
    "tipo_laboral",
    "salario_cliente",
    "total_otros_prestamos",
    "cuota_pactada",
    "puntaje_datacredito",
    "cant_creditosvigentes",
    "huella_consulta",
    "saldo_mora",
    "saldo_total",
    "saldo_principal",
    "saldo_mora_codeudor",
    "creditos_sectorFinanciero",
    "creditos_sectorCooperativo",
    "creditos_sectorReal",
    "promedio_ingresos_datacredito",
    "tendencia_ingresos",
    "id_prediccion",
    "fecha_prediccion_api",
    "prediccion_modelo",
    "resultado_prediccion",
    "probabilidad_incumplimiento",
    "probabilidad_pago_atiempo",
    "nivel_riesgo",
    "tiempo_prediccion_ms"
]


# Funciones de transformación

def crear_variables_derivadas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea las mismas variables derivadas utilizadas
    durante el feature engineering del modelo.
    """

    df = df.copy()

    # Conversión de fecha

    df["fecha_prestamo"] = pd.to_datetime(
        df["fecha_prestamo"],
        errors="coerce"
    )

    if df["fecha_prestamo"].isna().any():
        raise HTTPException(
            status_code=422,
            detail="La fecha_prestamo no tiene un formato válido."
        )

    # Variables temporales

    df["anio_prestamo"] = df["fecha_prestamo"].dt.year
    df["mes_prestamo"] = df["fecha_prestamo"].dt.month
    df["dia_prestamo"] = df["fecha_prestamo"].dt.day
    df["dia_semana_prestamo"] = df["fecha_prestamo"].dt.dayofweek
    df["semana_anio_prestamo"] = (
        df["fecha_prestamo"].dt.isocalendar().week.astype(int)
    )
    df["trimestre_prestamo"] = df["fecha_prestamo"].dt.quarter
    df["hora_prestamo"] = (
        df["fecha_prestamo"].dt.hour
    )
    df["es_fin_semana"] = (
        df["fecha_prestamo"].dt.dayofweek >= 5
    ).astype(int)

    # Variables financieras

    df["relacion_cuota_salario"] = (
        df["cuota_pactada"] /
        df["salario_cliente"].replace(0, 1)
    )

    df["relacion_deuda_salario"] = (
        df["total_otros_prestamos"] /
        df["salario_cliente"].replace(0, 1)
    )

    df["saldo_pendiente_estimado"] = (
        df["saldo_total"] -
        df["saldo_principal"]
    )

    return df


# Funciones auxiliares

def comprobar_modelo():
    """
    Comprueba que el modelo y el pipeline estén disponibles.
    """

    if modelo is None or pipeline_preprocesamiento is None:
        raise HTTPException(
            status_code=500,
            detail=(
                "El modelo o el pipeline no están disponibles. "
                "Revise las rutas de los archivos .joblib."
            )
        )


def convertir_registro_dataframe(
    registro: RegistroCredito
) -> pd.DataFrame:
    """
    Convierte el registro recibido en un DataFrame.
    """

    datos = registro.model_dump()

    return pd.DataFrame([datos])


def obtener_nivel_riesgo(
    probabilidad_incumplimiento: float
) -> str:
    """
    Clasifica el riesgo según la probabilidad de incumplimiento.
    """

    if probabilidad_incumplimiento < 0.40:
        return "Bajo"

    elif probabilidad_incumplimiento < 0.70:
        return "Medio"

    return "Alto"


def guardar_prediccion_csv(registro: dict):
    """
    Guarda una predicción en el archivo CSV.
    """

    registro_limpio = {
        columna: registro.get(columna, None)
        for columna in COLUMNAS_CSV
    }

    df_prediccion = pd.DataFrame(
        [registro_limpio],
        columns=COLUMNAS_CSV
    )

    archivo_existe = os.path.exists(ARCHIVO_PREDICCIONES)

    df_prediccion.to_csv(
        ARCHIVO_PREDICCIONES,
        mode="a",
        header=not archivo_existe,
        index=False,
        encoding="utf-8"
    )


# Función principal de predicción

def realizar_prediccion(registro: RegistroCredito) -> dict:
    """
    Ejecuta el proceso completo de predicción.
    """

    comprobar_modelo()

    inicio = time.perf_counter()

    # Convertir entrada a DataFrame

    df_entrada = convertir_registro_dataframe(registro)

    # Crear variables utilizadas durante el entrenamiento

    df_entrada = crear_variables_derivadas(df_entrada)

    # Transformar datos con el pipeline

    try:
        datos_transformados = pipeline_preprocesamiento.transform(
            df_entrada
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Error al transformar los datos: {str(error)}"
        )

    # Obtener predicción y probabilidades

    try:
        prediccion = modelo.predict(datos_transformados)[0]
        probabilidades = modelo.predict_proba(datos_transformados)[0]

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Error al realizar la predicción: {str(error)}"
        )

    # Identificar las clases del modelo

    clases_modelo = list(modelo.classes_)

    if 0 not in clases_modelo or 1 not in clases_modelo:
        raise HTTPException(
            status_code=500,
            detail="El modelo no contiene las clases esperadas 0 e 1."
        )

    indice_clase_0 = clases_modelo.index(0)
    indice_clase_1 = clases_modelo.index(1)

    probabilidad_incumplimiento = float(
        probabilidades[indice_clase_0]
    )

    probabilidad_pago_atiempo = float(
        probabilidades[indice_clase_1]
    )

    prediccion = int(prediccion)

    if prediccion == 1:
        resultado_prediccion = "Pago a tiempo"
    else:
        resultado_prediccion = "Posible incumplimiento"

    nivel_riesgo = obtener_nivel_riesgo(
        probabilidad_incumplimiento
    )

    tiempo_prediccion_ms = round(
        (time.perf_counter() - inicio) * 1000,
        4
    )

    id_prediccion = str(uuid.uuid4())

    fecha_prediccion_api = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    # Guardar registro original y resultado

    datos_originales = registro.model_dump()

    registro_guardar = {
        **datos_originales,
        "id_prediccion": id_prediccion,
        "fecha_prediccion_api": fecha_prediccion_api,
        "prediccion_modelo": prediccion,
        "resultado_prediccion": resultado_prediccion,
        "probabilidad_incumplimiento": probabilidad_incumplimiento,
        "probabilidad_pago_atiempo": probabilidad_pago_atiempo,
        "nivel_riesgo": nivel_riesgo,
        "tiempo_prediccion_ms": tiempo_prediccion_ms
    }

    guardar_prediccion_csv(registro_guardar)

    return {
        "id_prediccion": id_prediccion,
        "prediccion_modelo": prediccion,
        "resultado_prediccion": resultado_prediccion,
        "probabilidad_incumplimiento": round(
            probabilidad_incumplimiento,
            4
        ),
        "probabilidad_pago_atiempo": round(
            probabilidad_pago_atiempo,
            4
        ),
        "nivel_riesgo": nivel_riesgo,
        "tiempo_prediccion_ms": tiempo_prediccion_ms
    }


# Endpoints

@app.get("/")
def inicio():
    """
    Endpoint principal de la API.
    """

    return {
        "mensaje": "API de predicción de créditos funcionando.",
        "version": "1.2.0",
        "endpoints": [
            "/health",
            "/predict",
            "/predict/batch",
            "/docs"
        ]
    }


@app.get("/health")
def health():
    """
    Comprueba el estado de la API, el modelo y el pipeline.
    """

    modelo_disponible = modelo is not None
    pipeline_disponible = pipeline_preprocesamiento is not None

    return {
        "estado_api": "activa",
        "modelo_cargado": modelo_disponible,
        "pipeline_cargado": pipeline_disponible,
        "estado": (
            "correcto"
            if modelo_disponible and pipeline_disponible
            else "error"
        )
    }


@app.post(
    "/predict",
    response_model=ResultadoPrediccion
)
def predict(registro: RegistroCredito):
    """
    Realiza una predicción para un solo registro.
    """

    return realizar_prediccion(registro)


@app.post(
    "/predict/batch",
    response_model=RespuestaBatch
)
def predict_batch(solicitud: SolicitudBatch):
    """
    Realiza predicciones para múltiples registros.
    """

    resultados = []

    for registro in solicitud.registros:
        resultado = realizar_prediccion(registro)
        resultados.append(resultado)

    return {
        "cantidad_registros": len(resultados),
        "predicciones": resultados
    }


# Ejecución automática de la API

if __name__ == "__main__":
    print("Iniciando API de predicción de créditos...")
    print(
        "Documentación disponible en: "
        "http://127.0.0.1:8000/docs"
    )
    print(
        "Estado de la API disponible en: "
        "http://127.0.0.1:8000/health"
    )

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )