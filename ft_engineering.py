from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Rutas principales del proyecto
RUTA_PROYECTO = Path(__file__).resolve().parent

RUTA_DATOS = RUTA_PROYECTO / "Base_de_datos_preparada.csv"
RUTA_SALIDA = RUTA_PROYECTO / "resultados_feature_engineering"

TARGET = "Pago_atiempo"
TEST_SIZE = 0.20
RANDOM_STATE = 42

# Puntaje se excluye por su posible fuga de información
EXCLUIR_PUNTAJE = True


def imprimir_separador(titulo):
    """
    Imprime títulos para organizar la información en consola.
    """

    print("\n")
    print(titulo)
    print("-" * len(titulo))


def cargar_datos():
    """
    Carga el dataset preparado.
    """

    if not RUTA_DATOS.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo:\n{RUTA_DATOS}"
        )

    df = pd.read_csv(RUTA_DATOS)

    if df.empty:
        raise ValueError("El archivo de datos está vacío.")

    return df


def validar_datos(df):
    """
    Realiza validaciones básicas del dataset.
    """

    imprimir_separador("1. VALIDACIÓN INICIAL")

    if TARGET not in df.columns:
        raise ValueError(
            f"La variable objetivo '{TARGET}' no existe."
        )

    if df[TARGET].isnull().any():
        raise ValueError(
            f"La variable objetivo '{TARGET}' contiene valores nulos."
        )

    valores_target = sorted(df[TARGET].dropna().unique())

    if not set(valores_target).issubset({0, 1}):
        raise ValueError(
            f"La variable objetivo debe contener 0 y 1. "
            f"Valores encontrados: {valores_target}"
        )

    cantidad_duplicados = df.duplicated().sum()
    cantidad_nulos = df.isnull().sum().sum()

    print(f"{'Cantidad de registros:':<35}{df.shape[0]:>10,}")
    print(f"{'Cantidad de columnas:':<35}{df.shape[1]:>10,}")
    print(f"{'Registros duplicados:':<35}{cantidad_duplicados:>10,}")
    print(f"{'Valores nulos totales:':<35}{cantidad_nulos:>10,}")

    print("\nDistribución de la variable objetivo:")
    distribucion = df[TARGET].value_counts().sort_index()

    for valor, cantidad in distribucion.items():
        porcentaje = cantidad / len(df) * 100
        print(
            f"  Clase {valor}: {cantidad:>8,} registros "
            f"({porcentaje:>6.2f}%)"
        )


def preparar_variables(df):
    """
    Realiza ajustes y crea características derivadas.
    """

    imprimir_separador("2. PREPARACIÓN Y CREACIÓN DE CARACTERÍSTICAS")

    df = df.copy()

    # Convertir la fecha a formato datetime
    if "fecha_prestamo" in df.columns:
        df["fecha_prestamo"] = pd.to_datetime(
            df["fecha_prestamo"],
            errors="coerce"
        )

        # Crear características temporales
        df["anio_prestamo"] = (
            df["fecha_prestamo"].dt.year
        )

        df["mes_prestamo"] = (
            df["fecha_prestamo"].dt.month
        )

        df["dia_prestamo"] = (
            df["fecha_prestamo"].dt.day
        )

        df["dia_semana_prestamo"] = (
            df["fecha_prestamo"].dt.dayofweek
        )

        df["semana_anio_prestamo"] = (
            df["fecha_prestamo"].dt.isocalendar().week
        )

        df["trimestre_prestamo"] = (
            df["fecha_prestamo"].dt.quarter
        )

        df["hora_prestamo"] = (
            df["fecha_prestamo"].dt.hour
        )

        df["es_fin_semana"] = (
            df["dia_semana_prestamo"] >= 5
        ).astype(int)

        # La fecha original no se utiliza como categoría
        df = df.drop(columns=["fecha_prestamo"])

        print("Características temporales creadas:")
        print("  - anio_prestamo")
        print("  - mes_prestamo")
        print("  - dia_prestamo")
        print("  - dia_semana_prestamo")
        print("  - semana_anio_prestamo")
        print("  - trimestre_prestamo")
        print("  - hora_prestamo")
        print("  - es_fin_semana")

    # Crear relación entre cuota y salario si existen las columnas
    if {
        "cuota_pactada",
        "salario_cliente"
    }.issubset(df.columns):

        df["relacion_cuota_salario"] = np.where(
            df["salario_cliente"] > 0,
            df["cuota_pactada"] / df["salario_cliente"],
            np.nan
        )

        print(
            "  - relacion_cuota_salario "
            "(cuota pactada / salario)"
        )

    # Crear relación entre deuda y salario si existen las columnas
    if {
        "total_otros_prestamos",
        "salario_cliente"
    }.issubset(df.columns):

        df["relacion_deuda_salario"] = np.where(
            df["salario_cliente"] > 0,
            df["total_otros_prestamos"] / df["salario_cliente"],
            np.nan
        )

        print(
            "  - relacion_deuda_salario "
            "(otros préstamos / salario)"
        )

    # Crear saldo pendiente estimado
    if {
        "capital_prestado",
        "saldo_principal"
    }.issubset(df.columns):

        df["saldo_pendiente_estimado"] = np.where(
            df["capital_prestado"].notna(),
            df["capital_prestado"] - df["saldo_principal"],
            np.nan
        )

        print(
            "  - saldo_pendiente_estimado "
            "(capital prestado - saldo principal)"
        )

    # Convertir tipo_credito a categórica
    if "tipo_credito" in df.columns:
        df["tipo_credito"] = df["tipo_credito"].astype("string")

    return df


def separar_variables(df):
    """
    Separa las variables predictoras y la variable objetivo.
    """

    imprimir_separador("3. SEPARACIÓN DE VARIABLES")

    columnas_excluidas = [TARGET]

    if EXCLUIR_PUNTAJE and "puntaje" in df.columns:
        columnas_excluidas.append("puntaje")
        print(
            "Variable excluida: puntaje "
            "(posible fuga de información)"
        )

    X = df.drop(columns=columnas_excluidas)
    y = df[TARGET]

    # Identificar tipos de datos
    columnas_numericas = X.select_dtypes(
        include=["number", "bool"]
    ).columns.tolist()

    columnas_categoricas = X.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()

    print(f"\n{'Variables predictoras:':<35}{X.shape[1]:>10,}")
    print(f"{'Variables numéricas:':<35}{len(columnas_numericas):>10,}")
    print(f"{'Variables categóricas:':<35}{len(columnas_categoricas):>10,}")

    print("\nVariables numéricas:")
    for columna in columnas_numericas:
        print(f"  - {columna}")

    print("\nVariables categóricas:")
    for columna in columnas_categoricas:
        print(f"  - {columna}")

    return X, y, columnas_numericas, columnas_categoricas


def dividir_datos(X, y):
    """
    Divide los datos en entrenamiento y prueba.
    """

    imprimir_separador("4. DIVISIÓN DE LOS DATOS")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print(f"{'Registros de entrenamiento:':<35}{len(X_train):>10,}")
    print(f"{'Registros de prueba:':<35}{len(X_test):>10,}")
    print(f"{'Porcentaje de prueba:':<35}{TEST_SIZE * 100:>9.0f}%")
    print(f"{'Semilla aleatoria:':<35}{RANDOM_STATE:>10}")

    print("\nDistribución del objetivo en entrenamiento:")
    distribucion_train = y_train.value_counts().sort_index()

    for valor, cantidad in distribucion_train.items():
        porcentaje = cantidad / len(y_train) * 100
        print(
            f"  Clase {valor}: {cantidad:>8,} "
            f"({porcentaje:>6.2f}%)"
        )

    print("\nDistribución del objetivo en prueba:")
    distribucion_test = y_test.value_counts().sort_index()

    for valor, cantidad in distribucion_test.items():
        porcentaje = cantidad / len(y_test) * 100
        print(
            f"  Clase {valor}: {cantidad:>8,} "
            f"({porcentaje:>6.2f}%)"
        )

    return X_train, X_test, y_train, y_test


def construir_preprocesador(
    columnas_numericas,
    columnas_categoricas
):
    """
    Construye el pipeline de preprocesamiento.
    """

    imprimir_separador("5. CONSTRUCCIÓN DEL PREPROCESADOR")

    pipeline_numerico = Pipeline(
        steps=[
            (
                "imputador",
                SimpleImputer(strategy="median")
            ),
            (
                "escalador",
                StandardScaler()
            )
        ]
    )

    pipeline_categorico = Pipeline(
        steps=[
            (
                "imputador",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "codificador",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            )
        ]
    )

    preprocesador = ColumnTransformer(
        transformers=[
            (
                "numericas",
                pipeline_numerico,
                columnas_numericas
            ),
            (
                "categoricas",
                pipeline_categorico,
                columnas_categoricas
            )
        ],
        remainder="drop"
    )

    print("Transformaciones numéricas:")
    print("  - Imputación por mediana")
    print("  - Estandarización con StandardScaler")

    print("\nTransformaciones categóricas:")
    print("  - Imputación por categoría más frecuente")
    print("  - Codificación OneHotEncoder")
    print("  - Manejo de categorías desconocidas")

    return preprocesador


def aplicar_transformaciones(
    preprocesador,
    X_train,
    X_test
):
    """
    Ajusta el preprocesador con entrenamiento y transforma ambos conjuntos.
    """

    imprimir_separador("6. APLICACIÓN DE TRANSFORMACIONES")

    # Ajustar únicamente con entrenamiento
    X_train_transformado = preprocesador.fit_transform(X_train)

    # Transformar prueba con el preprocesador ya ajustado
    X_test_transformado = preprocesador.transform(X_test)

    print(
        f"{'Dimensión original de X_train:':<40}"
        f"{X_train.shape}"
    )

    print(
        f"{'Dimensión transformada de X_train:':<40}"
        f"{X_train_transformado.shape}"
    )

    print(
        f"{'Dimensión original de X_test:':<40}"
        f"{X_test.shape}"
    )

    print(
        f"{'Dimensión transformada de X_test:':<40}"
        f"{X_test_transformado.shape}"
    )

    print(
        "\nEl preprocesador se ajustó únicamente con "
        "los datos de entrenamiento."
    )

    return X_train_transformado, X_test_transformado


def obtener_nombres_caracteristicas(preprocesador):
    """
    Obtiene los nombres de las variables transformadas.
    """

    nombres = []

    for nombre_transformador, transformador, columnas in (
        preprocesador.transformers_
    ):
        if nombre_transformador == "remainder":
            continue

        if nombre_transformador == "numericas":
            nombres.extend(columnas)

        elif nombre_transformador == "categoricas":
            codificador = (
                transformador.named_steps["codificador"]
            )

            nombres_categoricos = (
                codificador.get_feature_names_out(columnas)
            )

            nombres.extend(nombres_categoricos)

    return nombres


def guardar_resultados(
    df_preparado,
    X_train_transformado,
    X_test_transformado,
    y_train,
    y_test,
    preprocesador,
    nombres_caracteristicas,
    columnas_numericas,
    columnas_categoricas
):
    """
    Guarda los resultados del proceso de ingeniería de características.
    """

    imprimir_separador("7. GUARDADO DE RESULTADOS")

    RUTA_SALIDA.mkdir(
        parents=True,
        exist_ok=True
    )

    nombres_caracteristicas = list(nombres_caracteristicas)

    X_train_df = pd.DataFrame(
        X_train_transformado,
        columns=nombres_caracteristicas
    )

    X_test_df = pd.DataFrame(
        X_test_transformado,
        columns=nombres_caracteristicas
    )

    y_train_df = y_train.reset_index(drop=True).to_frame()
    y_test_df = y_test.reset_index(drop=True).to_frame()

    # Guardar conjuntos transformados
    X_train_df.to_csv(
        RUTA_SALIDA / "X_train_transformado.csv",
        index=False
    )

    X_test_df.to_csv(
        RUTA_SALIDA / "X_test_transformado.csv",
        index=False
    )

    y_train_df.to_csv(
        RUTA_SALIDA / "y_train.csv",
        index=False
    )

    y_test_df.to_csv(
        RUTA_SALIDA / "y_test.csv",
        index=False
    )

    # Guardar el pipeline entrenado
    joblib.dump(
        preprocesador,
        RUTA_SALIDA / "pipeline_preprocesamiento.joblib"
    )

    # Guardar nombres de las características
    pd.DataFrame(
        {
            "indice": range(1, len(nombres_caracteristicas) + 1),
            "caracteristica": nombres_caracteristicas
        }
    ).to_csv(
        RUTA_SALIDA / "nombres_caracteristicas.csv",
        index=False
    )

    # Guardar resumen de nulos antes del procesamiento
    resumen_nulos = (
        df_preparado.isnull()
        .sum()
        .reset_index()
    )

    resumen_nulos.columns = [
        "variable",
        "cantidad_nulos"
    ]

    resumen_nulos["porcentaje_nulos"] = (
        resumen_nulos["cantidad_nulos"]
        / len(df_preparado)
        * 100
    )

    resumen_nulos = resumen_nulos.sort_values(
        by="cantidad_nulos",
        ascending=False
    )

    resumen_nulos.to_csv(
        RUTA_SALIDA / "resumen_valores_nulos.csv",
        index=False
    )

    # Guardar resumen de tipos de variables
    resumen_variables = pd.DataFrame(
        {
            "variable": df_preparado.columns,
            "tipo_dato": [
                str(tipo)
                for tipo in df_preparado.dtypes
            ],
            "cantidad_nulos": [
                df_preparado[columna].isnull().sum()
                for columna in df_preparado.columns
            ],
            "cantidad_unicos": [
                df_preparado[columna].nunique(
                    dropna=True
                )
                for columna in df_preparado.columns
            ]
        }
    )

    resumen_variables.to_csv(
        RUTA_SALIDA / "resumen_variables.csv",
        index=False
    )

    # Guardar resumen general del proceso
    resumen = [
        "RESUMEN DE INGENIERÍA DE CARACTERÍSTICAS",
        "",
        f"Archivo de entrada: {RUTA_DATOS}",
        f"Registros originales: {len(df_preparado):,}",
        f"Columnas después de preparación: {df_preparado.shape[1]:,}",
        f"Variable objetivo: {TARGET}",
        f"Registros de entrenamiento: {len(X_train_df):,}",
        f"Registros de prueba: {len(X_test_df):,}",
        f"Características finales: {X_train_df.shape[1]:,}",
        f"Porcentaje de prueba: {TEST_SIZE * 100:.0f}%",
        f"Semilla aleatoria: {RANDOM_STATE}",
        "",
        "VARIABLES NUMÉRICAS",
        *[f"- {columna}" for columna in columnas_numericas],
        "",
        "VARIABLES CATEGÓRICAS",
        *[f"- {columna}" for columna in columnas_categoricas],
        "",
        "TRANSFORMACIONES APLICADAS",
        "- Imputación numérica mediante la mediana",
        "- Estandarización de variables numéricas",
        "- Imputación categórica mediante la moda",
        "- Codificación OneHotEncoder",
        "- Manejo de categorías desconocidas",
        "- División estratificada entre entrenamiento y prueba",
        "- Ajuste del preprocesador únicamente con entrenamiento",
        "",
        "VARIABLES EXCLUIDAS",
        "- Pago_atiempo: variable objetivo",
        "- puntaje: excluida por posible fuga de información"
    ]

    with open(
        RUTA_SALIDA / "resumen_feature_engineering.txt",
        "w",
        encoding="utf-8"
    ) as archivo:
        archivo.write("\n".join(resumen))

    print(f"\n{'Archivos generados en:':<35}{RUTA_SALIDA}")
    print("\nArchivos creados:")

    archivos = sorted(RUTA_SALIDA.iterdir())

    for archivo in archivos:
        print(f"  - {archivo.name}")


def main():
    """
    Ejecuta el proceso completo.
    """

    print("Iniciando ingeniería de características")

    df = cargar_datos()

    validar_datos(df)

    df_preparado = preparar_variables(df)

    X, y, columnas_numericas, columnas_categoricas = (
        separar_variables(df_preparado)
    )

    X_train, X_test, y_train, y_test = dividir_datos(X, y)

    preprocesador = construir_preprocesador(
        columnas_numericas,
        columnas_categoricas
    )

    (
        X_train_transformado,
        X_test_transformado
    ) = aplicar_transformaciones(
        preprocesador,
        X_train,
        X_test
    )

    nombres_caracteristicas = (
        obtener_nombres_caracteristicas(preprocesador)
    )

    guardar_resultados(
        df_preparado=df_preparado,
        X_train_transformado=X_train_transformado,
        X_test_transformado=X_test_transformado,
        y_train=y_train,
        y_test=y_test,
        preprocesador=preprocesador,
        nombres_caracteristicas=nombres_caracteristicas,
        columnas_numericas=columnas_numericas,
        columnas_categoricas=columnas_categoricas
    )

    imprimir_separador("PROCESO FINALIZADO")

    print("La ingeniería de características terminó correctamente.")
    print(
        "Los datos están listos para utilizarse "
        "en el entrenamiento de modelos."
    )


if __name__ == "__main__":
    main()