import json
import urllib.request
import urllib.error

import pandas as pd


# Configuración

RUTA_BASE_DATOS = "Base_de_datos.csv"
URL_API = "http://127.0.0.1:8000/predict/batch"

CANTIDAD_REGISTROS = 100


# Columnas que necesita la API

COLUMNAS_API = [
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
    "tendencia_ingresos"
]


# Columnas numéricas

COLUMNAS_NUMERICAS = [
    "capital_prestado",
    "plazo_meses",
    "edad_cliente",
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
    "promedio_ingresos_datacredito"
]


# Columnas categóricas

COLUMNAS_CATEGORICAS = [
    "tipo_credito",
    "tipo_laboral",
    "tendencia_ingresos"
]


def preparar_registros():
    """
    Lee la base de datos y prepara los registros
    para enviarlos a la API.
    """

    print("Leyendo la base de datos...")

    df = pd.read_csv(RUTA_BASE_DATOS)

    print(
        f"Registros disponibles en la base: {len(df)}"
    )

    # Comprobar columnas requeridas

    columnas_faltantes = [
        columna
        for columna in COLUMNAS_API
        if columna not in df.columns
    ]

    if columnas_faltantes:
        raise ValueError(
            "Faltan estas columnas en la base de datos: "
            + str(columnas_faltantes)
        )

    # Seleccionar columnas para la API

    df_api = df[COLUMNAS_API].copy()

    # Seleccionar registros reproducibles

    cantidad = min(
        CANTIDAD_REGISTROS,
        len(df_api)
    )

    df_api = df_api.sample(
        n=cantidad,
        random_state=42
    ).copy()

    # Convertir y completar columnas numéricas

    for columna in COLUMNAS_NUMERICAS:
        df_api[columna] = pd.to_numeric(
            df_api[columna],
            errors="coerce"
        )

        mediana = df_api[columna].median()

        if pd.isna(mediana):
            mediana = 0

        df_api[columna] = df_api[columna].fillna(
            mediana
        )

    # Convertir y completar columnas categóricas

    for columna in COLUMNAS_CATEGORICAS:
        df_api[columna] = (
            df_api[columna]
            .fillna("Desconocido")
            .astype(str)
        )

    # Convertir la fecha con un formato definido

    df_api["fecha_prestamo"] = pd.to_datetime(
        df_api["fecha_prestamo"],
        errors="coerce"
    )

    # Si alguna fecha es inválida, utilizar una fecha válida

    fechas_validas = df_api["fecha_prestamo"].dropna()

    if fechas_validas.empty:
        raise ValueError(
            "No existen fechas válidas en los registros."
        )

    fecha_reemplazo = fechas_validas.iloc[0]

    df_api["fecha_prestamo"] = (
        df_api["fecha_prestamo"]
        .fillna(fecha_reemplazo)
        .dt.strftime("%Y-%m-%d")
    )

    # Convertir valores numéricos enteros

    columnas_enteras = [
        "plazo_meses",
        "edad_cliente",
        "cant_creditosvigentes",
        "huella_consulta",
        "creditos_sectorFinanciero",
        "creditos_sectorCooperativo",
        "creditos_sectorReal"
    ]

    for columna in columnas_enteras:
        df_api[columna] = (
            df_api[columna]
            .round()
            .astype(int)
        )

    # Reemplazar cualquier valor restante no válido

    df_api = df_api.astype(object).where(
        pd.notna(df_api),
        None
    )

    registros = df_api.to_dict(
        orient="records"
    )

    return registros


def enviar_a_api(registros):
    """
    Envía los registros al endpoint /predict/batch.
    """

    datos_solicitud = {
        "registros": registros
    }

    datos_json = json.dumps(
        datos_solicitud,
        ensure_ascii=False
    ).encode("utf-8")

    solicitud = urllib.request.Request(
        URL_API,
        data=datos_json,
        headers={
            "Content-Type": "application/json"
        },
        method="POST"
    )

    print("Enviando registros a la API...")

    try:
        with urllib.request.urlopen(solicitud) as respuesta:
            contenido = respuesta.read().decode(
                "utf-8"
            )

            resultado = json.loads(contenido)

            print(
                "Solicitud enviada correctamente."
            )

            print(
                "Cantidad de predicciones recibidas:",
                resultado["cantidad_registros"]
            )

            return resultado

    except urllib.error.HTTPError as error:
        detalle = error.read().decode("utf-8")

        print("La API respondió con un error.")
        print("Código:", error.code)
        print("Detalle:", detalle)

    except urllib.error.URLError as error:
        print("No fue posible conectarse con la API.")
        print(
            "Verifica que API.py esté ejecutándose."
        )
        print("Detalle:", error.reason)


if __name__ == "__main__":
    try:
        registros = preparar_registros()

        print(
            f"Se prepararon {len(registros)} "
            "registros para enviar."
        )

        enviar_a_api(registros)

    except Exception as error:
        print("Ocurrió un error:", error)