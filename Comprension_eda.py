from pathlib import Path
import shutil
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from tabulate import tabulate


warnings.filterwarnings("ignore")

sns.set_theme(
    style="whitegrid",
    context="notebook"
)


# Configuración de rutas

BASE_DIR = Path(__file__).resolve().parent

ARCHIVO_ENTRADA = BASE_DIR / "Base_de_datos_preparada.csv"
CARPETA_SALIDA = BASE_DIR / "resultados_eda_entrega"
CARPETA_GRAFICOS = CARPETA_SALIDA / "graficos"

ARCHIVO_EXCEL = CARPETA_SALIDA / "Reporte_EDA.xlsx"
ARCHIVO_TXT = CARPETA_SALIDA / "Reporte_EDA.txt"


# Configuración general

TARGET = "Pago_atiempo"

VARIABLES_CATEGORICAS = [
    "tipo_credito",
    "tipo_laboral",
    "tendencia_ingresos"
]

EXTENSIONES_IMAGENES = ["*.png", "*.jpg", "*.jpeg"]


# Funciones auxiliares

def limpiar_carpeta_salida():
    """
    Elimina los resultados anteriores y crea una estructura limpia.
    """

    if CARPETA_SALIDA.exists():
        shutil.rmtree(CARPETA_SALIDA)

    CARPETA_GRAFICOS.mkdir(
        parents=True,
        exist_ok=True
    )


def cargar_datos():
    """
    Carga el archivo preparado y ajusta los tipos de datos principales.
    """

    if not ARCHIVO_ENTRADA.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo: {ARCHIVO_ENTRADA}"
        )

    df = pd.read_csv(
        ARCHIVO_ENTRADA,
        low_memory=False
    )

    if "fecha_prestamo" in df.columns:
        df["fecha_prestamo"] = pd.to_datetime(
            df["fecha_prestamo"],
            errors="coerce"
        )

    if TARGET in df.columns:
        df[TARGET] = pd.to_numeric(
            df[TARGET],
            errors="coerce"
        )

        df[TARGET] = df[TARGET].astype("Int64")

    for columna in VARIABLES_CATEGORICAS:
        if columna in df.columns:
            df[columna] = df[columna].astype("string")

    return df


def formatear_entero(valor):
    """
    Formatea valores enteros sin mostrar decimales.
    """

    if pd.isna(valor):
        return ""

    try:
        return f"{int(round(float(valor))):,}".replace(",", ".")
    except (ValueError, TypeError):
        return str(valor)


def formatear_porcentaje(valor):
    """
    Formatea un valor como porcentaje.
    """

    if pd.isna(valor):
        return ""

    try:
        return f"{float(valor):,.2f}%".replace(",", "X").replace(".", ",").replace("X", ".")
    except (ValueError, TypeError):
        return str(valor)


def formatear_decimal(valor, decimales=2):
    """
    Formatea números decimales usando coma decimal.
    """

    if pd.isna(valor):
        return ""

    try:
        texto = f"{float(valor):,.{decimales}f}"
        return texto.replace(",", "X").replace(".", ",").replace("X", ".")
    except (ValueError, TypeError):
        return str(valor)


def formatear_dinero(valor):
    """
    Formatea valores monetarios sin decimales.
    """

    if pd.isna(valor):
        return ""

    try:
        texto = f"{float(valor):,.0f}"
        return texto.replace(",", "X").replace(".", ",").replace("X", ".")
    except (ValueError, TypeError):
        return str(valor)


def imprimir_tabla(df, titulo, max_filas=20):
    """
    Imprime tablas pequeñas y legibles en consola.
    """

    print(f"\n{titulo}")

    if df.empty:
        print("No hay información disponible.")
        return

    tabla = df.head(max_filas).copy()

    # Convertir las celdas a texto evita que tabulate
    # interprete conteos formateados como números decimales.
    for columna in tabla.columns:
        tabla[columna] = tabla[columna].apply(
            lambda valor: "" if pd.isna(valor) else str(valor)
        )

    print(
        tabulate(
            tabla,
            headers="keys",
            tablefmt="fancy_grid",
            showindex=False,
            stralign="left",
            numalign="right",
            disable_numparse=True
        )
    )

    if len(df) > max_filas:
        print(
            f"\nSe muestran {max_filas} registros de un total de {len(df):,}."
        )


def guardar_csv(df, nombre_archivo):
    """
    Guarda una tabla en formato CSV.
    """

    ruta = CARPETA_SALIDA / nombre_archivo

    df.to_csv(
        ruta,
        index=False,
        encoding="utf-8-sig"
    )


def ajustar_ancho_excel(writer):
    """
    Ajusta el ancho de las columnas y congela la primera fila de cada hoja.
    """

    for hoja in writer.book.worksheets:
        hoja.freeze_panes = "A2"
        hoja.auto_filter.ref = hoja.dimensions

        for columna in hoja.columns:
            max_longitud = 0
            letra_columna = columna[0].column_letter

            for celda in columna:
                valor = "" if celda.value is None else str(celda.value)
                max_longitud = max(
                    max_longitud,
                    len(valor)
                )

            ancho = min(
                max(max_longitud + 2, 12),
                35
            )

            hoja.column_dimensions[letra_columna].width = ancho


def guardar_excel(tablas):
    """
    Guarda todas las tablas principales en un único archivo Excel.
    """

    with pd.ExcelWriter(
        ARCHIVO_EXCEL,
        engine="openpyxl"
    ) as writer:

        for nombre_hoja, tabla in tablas.items():
            tabla.to_excel(
                writer,
                sheet_name=nombre_hoja,
                index=False
            )

        ajustar_ancho_excel(writer)


def guardar_reporte_txt(secciones):
    """
    Guarda un reporte de texto con una estructura clara.
    """

    with open(
        ARCHIVO_TXT,
        "w",
        encoding="utf-8"
    ) as archivo:

        for titulo, contenido in secciones:
            archivo.write(f"{titulo}\n")
            archivo.write(f"{contenido}\n\n")


def crear_grafico_distribuciones_numericas(df, variables_numericas):
    """
    Genera un gráfico con las distribuciones de las variables numéricas.
    """

    variables = variables_numericas[:12]

    if not variables:
        return

    cantidad = len(variables)
    columnas = 3
    filas = int(np.ceil(cantidad / columnas))

    fig, ejes = plt.subplots(
        filas,
        columnas,
        figsize=(18, 5 * filas)
    )

    ejes = np.array(ejes).reshape(-1)

    for indice, variable in enumerate(variables):
        serie = pd.to_numeric(
            df[variable],
            errors="coerce"
        ).dropna()

        if serie.empty:
            ejes[indice].set_visible(False)
            continue

        ejes[indice].hist(
            serie,
            bins=30
        )

        ejes[indice].set_title(
            f"Distribución de {variable}"
        )
        ejes[indice].set_xlabel(variable)
        ejes[indice].set_ylabel("Frecuencia")

    for indice in range(len(variables), len(ejes)):
        ejes[indice].set_visible(False)

    fig.suptitle(
        "Distribución de variables numéricas",
        fontsize=16
    )

    fig.tight_layout()

    fig.savefig(
        CARPETA_GRAFICOS / "01_distribuciones_numericas.png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)


def crear_grafico_boxplots(df, variables_numericas):
    """
    Genera boxplots para revisar dispersión y valores atípicos.
    """

    variables = variables_numericas[:12]

    if not variables:
        return

    datos = df[variables].copy()

    for columna in datos.columns:
        datos[columna] = pd.to_numeric(
            datos[columna],
            errors="coerce"
        )

    datos = datos.dropna(
        axis=1,
        how="all"
    )

    if datos.empty:
        return

    fig, ax = plt.subplots(
        figsize=(16, 8)
    )

    datos.boxplot(
        ax=ax,
        rot=75
    )

    ax.set_title(
        "Boxplots de variables numéricas"
    )
    ax.set_xlabel("Variables")
    ax.set_ylabel("Valores")

    fig.tight_layout()

    fig.savefig(
        CARPETA_GRAFICOS / "02_boxplots_numericos.png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)


def crear_grafico_target(df):
    """
    Genera la distribución de la variable objetivo.
    """

    if TARGET not in df.columns:
        return

    conteos = (
        df[TARGET]
        .value_counts(dropna=False)
        .sort_index()
    )

    etiquetas = [
        "No paga a tiempo" if valor == 0
        else "Paga a tiempo" if valor == 1
        else "Nulo"
        for valor in conteos.index
    ]

    valores = conteos.values

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    barras = ax.bar(
        etiquetas,
        valores
    )

    ax.set_title(
        "Distribución de Pago_atiempo"
    )
    ax.set_xlabel("Categoría")
    ax.set_ylabel("Cantidad de registros")

    for barra, valor in zip(barras, valores):
        ax.text(
            barra.get_x() + barra.get_width() / 2,
            barra.get_height(),
            formatear_entero(valor),
            ha="center",
            va="bottom"
        )

    fig.tight_layout()

    fig.savefig(
        CARPETA_GRAFICOS / "03_distribucion_target.png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)


def crear_grafico_frecuencias_categoricas(df, variables_categoricas):
    """
    Genera gráficos de frecuencia para las variables categóricas.
    """

    variables = [
        variable
        for variable in variables_categoricas
        if variable in df.columns
    ]

    if not variables:
        return

    cantidad = len(variables)

    fig, ejes = plt.subplots(
        cantidad,
        1,
        figsize=(14, 5 * cantidad)
    )

    if cantidad == 1:
        ejes = [ejes]

    for eje, variable in zip(ejes, variables):
        conteos = (
            df[variable]
            .fillna("Nulo")
            .value_counts()
            .head(15)
        )

        conteos.sort_values().plot(
            kind="barh",
            ax=eje
        )

        eje.set_title(
            f"Frecuencia de {variable}"
        )
        eje.set_xlabel("Cantidad de registros")
        eje.set_ylabel(variable)

    fig.tight_layout()

    fig.savefig(
        CARPETA_GRAFICOS / "04_frecuencias_categoricas.png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)


def crear_grafico_numericas_por_target(df, variables_numericas):
    """
    Compara variables numéricas según el valor del target.
    """

    if TARGET not in df.columns:
        return

    variables = variables_numericas[:6]

    if not variables:
        return

    cantidad = len(variables)
    columnas = 2
    filas = int(np.ceil(cantidad / columnas))

    fig, ejes = plt.subplots(
        filas,
        columnas,
        figsize=(16, 5 * filas)
    )

    ejes = np.array(ejes).reshape(-1)

    for indice, variable in enumerate(variables):
        datos = df[[TARGET, variable]].copy()

        datos[variable] = pd.to_numeric(
            datos[variable],
            errors="coerce"
        )

        datos = datos.dropna()

        if datos.empty:
            ejes[indice].set_visible(False)
            continue

        sns.boxplot(
            data=datos,
            x=TARGET,
            y=variable,
            ax=ejes[indice]
        )

        ejes[indice].set_title(
            f"{variable} según {TARGET}"
        )
        ejes[indice].set_xlabel("Pago_atiempo")
        ejes[indice].set_ylabel(variable)

    for indice in range(len(variables), len(ejes)):
        ejes[indice].set_visible(False)

    fig.suptitle(
        "Comparación de variables numéricas según el target",
        fontsize=16
    )

    fig.tight_layout()

    fig.savefig(
        CARPETA_GRAFICOS / "05_numericas_por_target.png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)


def crear_grafico_categoricas_por_target(df, variables_categoricas):
    """
    Muestra la distribución porcentual del target dentro de cada categoría.
    """

    if TARGET not in df.columns:
        return

    variables = [
        variable
        for variable in variables_categoricas
        if variable in df.columns
    ]

    if not variables:
        return

    cantidad = len(variables)

    fig, ejes = plt.subplots(
        cantidad,
        1,
        figsize=(15, 5 * cantidad)
    )

    if cantidad == 1:
        ejes = [ejes]

    for eje, variable in zip(ejes, variables):
        tabla = pd.crosstab(
            df[variable].fillna("Nulo"),
            df[TARGET],
            normalize="index"
        ) * 100

        tabla = tabla.head(15)

        tabla.plot(
            kind="bar",
            stacked=True,
            ax=eje
        )

        eje.set_title(
            f"Comportamiento de {TARGET} según {variable}"
        )
        eje.set_xlabel(variable)
        eje.set_ylabel("Porcentaje")
        eje.legend(
            title=TARGET,
            loc="upper right"
        )

        eje.tick_params(
            axis="x",
            rotation=45
        )

    fig.tight_layout()

    fig.savefig(
        CARPETA_GRAFICOS / "06_categoricas_por_target.png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)


def crear_grafico_correlacion(matriz_correlacion):
    """
    Genera un mapa de calor de la matriz de correlación.
    """

    if matriz_correlacion.empty:
        return

    fig, ax = plt.subplots(
        figsize=(18, 14)
    )

    sns.heatmap(
        matriz_correlacion,
        cmap="coolwarm",
        center=0,
        annot=False,
        linewidths=0.3,
        ax=ax
    )

    ax.set_title(
        "Matriz de correlación de variables numéricas"
    )

    fig.tight_layout()

    fig.savefig(
        CARPETA_GRAFICOS / "07_matriz_correlacion.png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)


def crear_grafico_relaciones_financieras(df):
    """
    Genera relaciones visuales entre variables financieras.
    """

    # Nombres reales de las variables financieras presentes en la base.
    # Se utilizan alternativas para que el gráfico no se omita por
    # buscar nombres que no existen en el archivo preparado.
    posibles_variables = [
        "salario_cliente",
        "capital_prestado",
        "cuota_pactada",
        "total_otros_prestamos",
        "saldo_total",
        "saldo_principal",
        "saldo_pendiente_estimado"
    ]

    variables = [
        variable
        for variable in posibles_variables
        if variable in df.columns
    ]

    variables = [
        variable
        for variable in variables
        if variable in df.columns
    ]

    if len(variables) < 2:
        return

    datos = df[variables].copy()

    for columna in datos.columns:
        datos[columna] = pd.to_numeric(
            datos[columna],
            errors="coerce"
        )

    datos = datos.dropna()

    if datos.empty:
        return

    datos = datos.head(3000)

    fig, ax = plt.subplots(
        figsize=(12, 8)
    )

    primera = variables[0]
    segunda = variables[1]

    ax.scatter(
        datos[primera],
        datos[segunda],
        alpha=0.35
    )

    ax.set_title(
        f"Relación entre {primera} y {segunda}"
    )
    ax.set_xlabel(primera)
    ax.set_ylabel(segunda)

    fig.tight_layout()

    fig.savefig(
        CARPETA_GRAFICOS / "08_relaciones_financieras.png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)


def crear_grafico_puntajes(df):
    """
    Analiza visualmente el comportamiento de variables relacionadas con puntajes.
    """

    posibles_variables = [
        "puntaje",
        "puntaje_datacredito",
        "score_datacredito",
        "puntaje_crediticio",
        "calificacion_crediticia"
    ]

    variables = [
        variable
        for variable in posibles_variables
        if variable in df.columns
    ]

    if not variables or TARGET not in df.columns:
        return

    variable = variables[0]

    datos = df[[TARGET, variable]].copy()

    datos[variable] = pd.to_numeric(
        datos[variable],
        errors="coerce"
    )

    datos = datos.dropna()

    if datos.empty:
        return

    fig, ax = plt.subplots(
        figsize=(10, 7)
    )

    sns.boxplot(
        data=datos,
        x=TARGET,
        y=variable,
        ax=ax
    )

    ax.set_title(
        f"Comportamiento de {variable} según {TARGET}"
    )
    ax.set_xlabel("Pago_atiempo")
    ax.set_ylabel(variable)

    fig.tight_layout()

    fig.savefig(
        CARPETA_GRAFICOS / "09_comportamiento_puntajes.png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)


def crear_grafico_temporal(df):
    """
    Muestra la cantidad de préstamos por año y mes.
    """

    if "fecha_prestamo" not in df.columns:
        return

    fechas = df["fecha_prestamo"].dropna()

    if fechas.empty:
        return

    temporal = (
        fechas
        .dt.to_period("M")
        .value_counts()
        .sort_index()
    )

    if temporal.empty:
        return

    fig, ax = plt.subplots(
        figsize=(15, 7)
    )

    ax.plot(
        temporal.index.astype(str),
        temporal.values,
        marker="o"
    )

    ax.set_title(
        "Cantidad de préstamos por mes"
    )
    ax.set_xlabel("Mes")
    ax.set_ylabel("Cantidad de préstamos")

    ax.tick_params(
        axis="x",
        rotation=75
    )

    fig.tight_layout()

    fig.savefig(
        CARPETA_GRAFICOS / "10_comportamiento_temporal.png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)


# Construcción de tablas de análisis

def construir_resumen_inicial(df):
    """
    Construye el resumen general del dataset.
    """

    filas = [
        {
            "indicador": "Registros",
            "valor": len(df)
        },
        {
            "indicador": "Columnas",
            "valor": len(df.columns)
        },
        {
            "indicador": "Celdas totales",
            "valor": df.shape[0] * df.shape[1]
        },
        {
            "indicador": "Valores nulos",
            "valor": int(df.isna().sum().sum())
        },
        {
            "indicador": "Porcentaje total de nulos",
            "valor": (
                df.isna().sum().sum()
                / (df.shape[0] * df.shape[1])
            ) * 100
        },
        {
            "indicador": "Variables numéricas",
            "valor": len(df.select_dtypes(include=np.number).columns)
        },
        {
            "indicador": "Variables categóricas",
            "valor": len(
                df.select_dtypes(
                    include=["object", "string", "category"]
                ).columns
            )
        },
        {
            "indicador": "Variables de fecha",
            "valor": len(
                df.select_dtypes(
                    include=["datetime", "datetimetz"]
                ).columns
            )
        }
    ]

    resumen = pd.DataFrame(filas)

    return resumen


def construir_calidad_datos(df):
    """
    Construye el análisis de calidad por columna.
    """

    filas = []

    for columna in df.columns:
        nulos = int(df[columna].isna().sum())
        registros = len(df)
        unicos = int(df[columna].nunique(dropna=True))

        filas.append(
            {
                "variable": columna,
                "tipo_dato": str(df[columna].dtype),
                "registros": registros,
                "valores_nulos": nulos,
                "porcentaje_nulos": (
                    nulos / registros * 100
                    if registros > 0
                    else 0
                ),
                "valores_unicos": unicos,
                "porcentaje_unicos": (
                    unicos / registros * 100
                    if registros > 0
                    else 0
                )
            }
        )

    calidad = pd.DataFrame(filas)

    calidad = calidad.sort_values(
        by="porcentaje_nulos",
        ascending=False
    ).reset_index(drop=True)

    return calidad


def construir_clasificacion_variables(df):
    """
    Clasifica las variables según su tipo y función analítica.
    """

    filas = []

    for columna in df.columns:
        tipo = str(df[columna].dtype)

        if columna == TARGET:
            clasificacion = "Variable objetivo"
        elif columna in VARIABLES_CATEGORICAS:
            clasificacion = "Categórica"
        elif pd.api.types.is_datetime64_any_dtype(df[columna]):
            clasificacion = "Fecha"
        elif pd.api.types.is_numeric_dtype(df[columna]):
            clasificacion = "Numérica"
        else:
            clasificacion = "Texto u otro tipo"

        filas.append(
            {
                "variable": columna,
                "tipo_dato": tipo,
                "clasificacion": clasificacion,
                "valores_nulos": int(df[columna].isna().sum()),
                "valores_unicos": int(df[columna].nunique(dropna=True))
            }
        )

    return pd.DataFrame(filas)


def construir_target(df):
    """
    Construye el resumen de distribución del target.
    """

    if TARGET not in df.columns:
        return pd.DataFrame(
            columns=[
                "categoria",
                "cantidad",
                "porcentaje"
            ]
        )

    conteos = (
        df[TARGET]
        .value_counts(dropna=False)
        .sort_index()
    )

    total = len(df)

    filas = []

    for categoria, cantidad in conteos.items():
        if pd.isna(categoria):
            nombre = "Nulo"
        elif categoria == 0:
            nombre = "No paga a tiempo"
        elif categoria == 1:
            nombre = "Paga a tiempo"
        else:
            nombre = str(categoria)

        filas.append(
            {
                "categoria": nombre,
                "cantidad": int(cantidad),
                "porcentaje": (
                    cantidad / total * 100
                    if total > 0
                    else 0
                )
            }
        )

    return pd.DataFrame(filas)


def construir_estadisticas_numericas(df):
    """
    Construye estadísticas descriptivas en formato vertical.
    """

    variables_numericas = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    filas = []

    for variable in variables_numericas:
        serie = pd.to_numeric(
            df[variable],
            errors="coerce"
        ).dropna()

        if serie.empty:
            continue

        filas.append(
            {
                "variable": variable,
                "registros_validos": int(serie.count()),
                "media": serie.mean(),
                "mediana": serie.median(),
                "desviacion_estandar": serie.std(),
                "minimo": serie.min(),
                "percentil_25": serie.quantile(0.25),
                "percentil_75": serie.quantile(0.75),
                "maximo": serie.max(),
                "asimetria": serie.skew()
            }
        )

    return pd.DataFrame(filas)


def construir_frecuencias_categoricas(df):
    """
    Construye una tabla consolidada de frecuencias categóricas.
    """

    filas = []

    for variable in VARIABLES_CATEGORICAS:
        if variable not in df.columns:
            continue

        serie = df[variable].fillna("Nulo")

        conteos = serie.value_counts(
            dropna=False
        )

        total = len(serie)

        for categoria, cantidad in conteos.items():
            filas.append(
                {
                    "variable": variable,
                    "categoria": str(categoria),
                    "cantidad": int(cantidad),
                    "porcentaje": (
                        cantidad / total * 100
                        if total > 0
                        else 0
                    )
                }
            )

    return pd.DataFrame(filas)


def construir_comparacion_numerica_target(df):
    """
    Compara las variables numéricas según cada categoría del target.
    """

    if TARGET not in df.columns:
        return pd.DataFrame()

    variables_numericas = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    variables_numericas = [
        variable
        for variable in variables_numericas
        if variable != TARGET
    ]

    filas = []

    for variable in variables_numericas:
        datos = df[[TARGET, variable]].copy()

        datos[variable] = pd.to_numeric(
            datos[variable],
            errors="coerce"
        )

        datos = datos.dropna(
            subset=[TARGET, variable]
        )

        grupo_0 = datos.loc[
            datos[TARGET] == 0,
            variable
        ]

        grupo_1 = datos.loc[
            datos[TARGET] == 1,
            variable
        ]

        media_0 = grupo_0.mean() if not grupo_0.empty else np.nan
        media_1 = grupo_1.mean() if not grupo_1.empty else np.nan

        diferencia = (
            media_1 - media_0
            if not pd.isna(media_0) and not pd.isna(media_1)
            else np.nan
        )

        filas.append(
            {
                "variable": variable,
                "n_pago_0": int(grupo_0.count()),
                "media_pago_0": media_0,
                "mediana_pago_0": grupo_0.median(),
                "n_pago_1": int(grupo_1.count()),
                "media_pago_1": media_1,
                "mediana_pago_1": grupo_1.median(),
                "diferencia_media_pago_1_menos_pago_0": diferencia
            }
        )

    resultado = pd.DataFrame(filas)

    return resultado


def construir_comportamiento_categoricas_target(df):
    """
    Analiza el comportamiento del target dentro de cada categoría.
    """

    if TARGET not in df.columns:
        return pd.DataFrame()

    filas = []

    for variable in VARIABLES_CATEGORICAS:
        if variable not in df.columns:
            continue

        datos = df[[variable, TARGET]].copy()
        datos[variable] = datos[variable].fillna("Nulo")

        agrupado = (
            datos
            .groupby([variable, TARGET], dropna=False)
            .size()
            .reset_index(name="cantidad")
        )

        totales_categoria = (
            datos
            .groupby(variable, dropna=False)
            .size()
            .reset_index(name="total_categoria")
        )

        agrupado = agrupado.merge(
            totales_categoria,
            on=variable,
            how="left"
        )

        agrupado["porcentaje_categoria"] = (
            agrupado["cantidad"]
            / agrupado["total_categoria"]
            * 100
        )

        for _, fila in agrupado.iterrows():
            categoria = str(fila[variable])
            target = fila[TARGET]

            if target == 0:
                nombre_target = "No paga a tiempo"
            elif target == 1:
                nombre_target = "Paga a tiempo"
            else:
                nombre_target = "Nulo"

            filas.append(
                {
                    "variable": variable,
                    "categoria": categoria,
                    "target": nombre_target,
                    "cantidad": int(fila["cantidad"]),
                    "total_categoria": int(fila["total_categoria"]),
                    "porcentaje_dentro_categoria": fila[
                        "porcentaje_categoria"
                    ]
                }
            )

    return pd.DataFrame(filas)


def construir_correlaciones(df):
    """
    Construye las correlaciones entre las variables numéricas.
    """

    variables_numericas = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    matriz = df[variables_numericas].corr(
        numeric_only=True
    )

    filas = []

    for variable_1 in matriz.columns:
        for variable_2 in matriz.columns:
            if variable_1 >= variable_2:
                continue

            correlacion = matriz.loc[
                variable_1,
                variable_2
            ]

            filas.append(
                {
                    "variable_1": variable_1,
                    "variable_2": variable_2,
                    "correlacion": correlacion,
                    "correlacion_absoluta": abs(correlacion)
                }
            )

    correlaciones = pd.DataFrame(filas)

    if not correlaciones.empty:
        correlaciones = correlaciones.sort_values(
            by="correlacion_absoluta",
            ascending=False
        ).reset_index(drop=True)

    return matriz, correlaciones


def construir_correlaciones_target(df):
    """
    Calcula la correlación de las variables numéricas con el target.
    """

    if TARGET not in df.columns:
        return pd.DataFrame()

    variables_numericas = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    if TARGET not in variables_numericas:
        return pd.DataFrame()

    correlaciones = (
        df[variables_numericas]
        .corr(numeric_only=True)[TARGET]
        .drop(labels=[TARGET], errors="ignore")
        .sort_values(
            key=lambda serie: serie.abs(),
            ascending=False
        )
    )

    resultado = pd.DataFrame(
        {
            "variable": correlaciones.index,
            "correlacion_con_target": correlaciones.values,
            "correlacion_absoluta": correlaciones.abs().values
        }
    )

    return resultado


def construir_hallazgos(
    df,
    calidad,
    target,
    correlaciones_target
):
    """
    Genera hallazgos descriptivos basados en los resultados del EDA.
    """

    hallazgos = []

    total_registros = len(df)

    if TARGET in df.columns:
        distribucion = df[TARGET].value_counts(
            dropna=False
        )

        cantidad_0 = int(distribucion.get(0, 0))
        cantidad_1 = int(distribucion.get(1, 0))

        porcentaje_0 = (
            cantidad_0 / total_registros * 100
            if total_registros > 0
            else 0
        )

        porcentaje_1 = (
            cantidad_1 / total_registros * 100
            if total_registros > 0
            else 0
        )

        hallazgos.append(
            "La variable objetivo presenta un desbalance importante: "
            f"{formatear_entero(cantidad_1)} registros corresponden a pagos "
            f"a tiempo ({formatear_porcentaje(porcentaje_1)}) y "
            f"{formatear_entero(cantidad_0)} corresponden a pagos fuera "
            f"de tiempo ({formatear_porcentaje(porcentaje_0)})."
        )

        hallazgos.append(
            "Debido al desbalance del target, en una etapa posterior de "
            "modelado no se debería utilizar únicamente la exactitud "
            "como métrica de evaluación. También deberían considerarse "
            "precision, recall, F1-score, matriz de confusión y ROC-AUC."
        )

    variables_con_nulos = calidad[
        calidad["valores_nulos"] > 0
    ].copy()

    if not variables_con_nulos.empty:
        principales_nulos = variables_con_nulos.head(5)

        texto_nulos = []

        for _, fila in principales_nulos.iterrows():
            texto_nulos.append(
                f"{fila['variable']} "
                f"({formatear_porcentaje(fila['porcentaje_nulos'])})"
            )

        hallazgos.append(
            "Las variables con mayor proporción de valores nulos son: "
            + ", ".join(texto_nulos)
            + "."
        )

        hallazgos.append(
            "Los valores nulos deben tratarse dentro del pipeline de "
            "preprocesamiento del modelo. La estrategia dependerá del "
            "tipo de variable y del significado de los datos faltantes."
        )

    if not correlaciones_target.empty:
        principales = correlaciones_target.head(5)

        texto_correlaciones = []

        for _, fila in principales.iterrows():
            texto_correlaciones.append(
                f"{fila['variable']} "
                f"({formatear_decimal(fila['correlacion_con_target'], 4)})"
            )

        hallazgos.append(
            "Las variables con mayor correlación absoluta con "
            f"{TARGET} son: "
            + ", ".join(texto_correlaciones)
            + "."
        )

        correlacion_mayor = correlaciones_target.iloc[0]

        if (
            abs(correlacion_mayor["correlacion_con_target"])
            >= 0.80
        ):
            hallazgos.append(
                "Se observa al menos una correlación muy elevada entre "
                f"una variable numérica y {TARGET}. La variable "
                f"{correlacion_mayor['variable']} presenta una correlación "
                f"de {formatear_decimal(correlacion_mayor['correlacion_con_target'], 4)}. "
                "Esta relación debe revisarse para descartar fuga de "
                "información o variables construidas utilizando datos "
                "posteriores al momento de otorgamiento del crédito."
            )

    nombres_columnas = set(df.columns)

    if "relacion_cuota_salario" in nombres_columnas:
        cuota_salario = pd.to_numeric(
            df["relacion_cuota_salario"],
            errors="coerce"
        )

        cantidad_mayor_uno = int(
            (cuota_salario > 1).sum()
        )

        hallazgos.append(
            "La variable relacion_cuota_salario permite identificar "
            f"{formatear_entero(cantidad_mayor_uno)} registros en los que "
            "la cuota mensual supera el ingreso utilizado en el cálculo. "
            "Estos casos deben revisarse desde el punto de vista financiero "
            "y de calidad de datos."
        )

    if "relacion_deuda_salario" in nombres_columnas:
        deuda_salario = pd.to_numeric(
            df["relacion_deuda_salario"],
            errors="coerce"
        )

        cantidad_mayor_uno = int(
            (deuda_salario > 1).sum()
        )

        hallazgos.append(
            "La variable relacion_deuda_salario presenta "
            f"{formatear_entero(cantidad_mayor_uno)} registros con valores "
            "superiores a 1. Esto indica que la deuda supera el ingreso "
            "considerado y puede representar una señal de presión financiera."
        )

    hallazgos.append(
        "Las variables financieras pueden presentar asimetría y valores "
        "atípicos. En esta etapa exploratoria no se eliminaron ni "
        "modificaron estos registros, ya que primero debe determinarse "
        "si representan errores o situaciones reales."
    )

    if "tipo_credito" in df.columns:
        hallazgos.append(
            "tipo_credito debe tratarse como una variable categórica "
            "nominal si sus valores representan categorías de crédito. "
            "No debe interpretarse automáticamente como una escala numérica."
        )

    hallazgos.append(
        "Los resultados del EDA son descriptivos y no permiten afirmar "
        "causalidad. Las relaciones encontradas deben validarse durante "
        "la preparación del modelo y mediante pruebas estadísticas o "
        "técnicas de validación adicionales."
    )

    return pd.DataFrame(
        {
            "numero": range(1, len(hallazgos) + 1),
            "hallazgo": hallazgos
        }
    )

def mostrar_estadisticas_numericas(estadisticas):
    """
    Muestra estadísticas descriptivas de las variables numéricas.
    """

    if estadisticas.empty:
        print("\nNo hay estadísticas numéricas disponibles.")
        return

    tabla = estadisticas.copy()

    columnas = [
        "variable",
        "registros_validos",
        "media",
        "mediana",
        "desviacion_estandar",
        "minimo",
        "percentil_25",
        "percentil_75",
        "maximo"
    ]

    columnas = [
        columna
        for columna in columnas
        if columna in tabla.columns
    ]

    tabla = tabla[columnas].copy()

    tabla["registros_validos"] = tabla["registros_validos"].map(
        formatear_entero
    )

    columnas_numericas = [
        columna
        for columna in columnas
        if columna != "variable" and columna != "registros_validos"
    ]

    for columna in columnas_numericas:
        tabla[columna] = tabla[columna].map(
            lambda valor: formatear_decimal(valor, 2)
        )

    imprimir_tabla(
        tabla,
        "Estadísticas descriptivas de variables numéricas",
        max_filas=30
    )

def mostrar_frecuencias_categoricas(frecuencias):
    """
    Muestra las frecuencias de las variables categóricas.
    """

    if frecuencias.empty:
        print("\nNo hay frecuencias categóricas disponibles.")
        return

    tabla = frecuencias.copy()

    tabla["cantidad"] = tabla["cantidad"].map(
        formatear_entero
    )

    tabla["porcentaje"] = tabla["porcentaje"].map(
        formatear_porcentaje
    )

    imprimir_tabla(
        tabla[
            [
                "variable",
                "categoria",
                "cantidad",
                "porcentaje"
            ]
        ],
        "Frecuencias de variables categóricas",
        max_filas=30
    )

def mostrar_comparacion_numerica_target(comparacion):
    """
    Muestra la comparación de variables numéricas entre los grupos del target.
    """

    if comparacion.empty:
        print("\nNo hay comparación numérica disponible.")
        return

    tabla = comparacion.copy()

    tabla["n_pago_0"] = tabla["n_pago_0"].map(
        formatear_entero
    )

    tabla["n_pago_1"] = tabla["n_pago_1"].map(
        formatear_entero
    )

    columnas_decimales = [
        "media_pago_0",
        "mediana_pago_0",
        "media_pago_1",
        "mediana_pago_1",
        "diferencia_media_pago_1_menos_pago_0"
    ]

    for columna in columnas_decimales:
        if columna in tabla.columns:
            tabla[columna] = tabla[columna].map(
                lambda valor: formatear_decimal(valor, 2)
            )

    imprimir_tabla(
        tabla,
        "Comparación de variables numéricas según Pago_atiempo",
        max_filas=30
    )

def mostrar_comportamiento_categoricas_target(comportamiento):
    """
    Muestra cómo se comporta el target dentro de cada categoría.
    """

    if comportamiento.empty:
        print("\nNo hay análisis categórico frente al target disponible.")
        return

    tabla = comportamiento.copy()

    tabla["cantidad"] = tabla["cantidad"].map(
        formatear_entero
    )

    tabla["total_categoria"] = tabla["total_categoria"].map(
        formatear_entero
    )

    tabla["porcentaje_dentro_categoria"] = (
        tabla["porcentaje_dentro_categoria"].map(
            formatear_porcentaje
        )
    )

    imprimir_tabla(
        tabla[
            [
                "variable",
                "categoria",
                "target",
                "cantidad",
                "total_categoria",
                "porcentaje_dentro_categoria"
            ]
        ],
        "Comportamiento de variables categóricas según Pago_atiempo",
        max_filas=40
    )

def mostrar_correlaciones(correlaciones):
    """
    Muestra las relaciones más fuertes entre variables numéricas.
    """

    if correlaciones.empty:
        print("\nNo hay correlaciones disponibles.")
        return

    tabla = correlaciones.copy()

    tabla["correlacion"] = tabla["correlacion"].map(
        lambda valor: formatear_decimal(valor, 4)
    )

    tabla["correlacion_absoluta"] = tabla["correlacion_absoluta"].map(
        lambda valor: formatear_decimal(valor, 4)
    )

    imprimir_tabla(
        tabla[
            [
                "variable_1",
                "variable_2",
                "correlacion",
                "correlacion_absoluta"
            ]
        ],
        "Correlaciones más fuertes entre variables numéricas",
        max_filas=20
    )

def construir_resumen_outliers(df):
    """
    Identifica posibles valores atípicos mediante el rango intercuartílico.
    No elimina ni modifica registros.
    """

    variables_numericas = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    variables_numericas = [
        variable
        for variable in variables_numericas
        if variable != TARGET
    ]

    filas = []

    for variable in variables_numericas:
        serie = pd.to_numeric(
            df[variable],
            errors="coerce"
        ).dropna()

        if serie.empty:
            continue

        q1 = serie.quantile(0.25)
        q3 = serie.quantile(0.75)
        iqr = q3 - q1

        limite_inferior = q1 - 1.5 * iqr
        limite_superior = q3 + 1.5 * iqr

        cantidad_outliers = int(
            (
                (serie < limite_inferior)
                | (serie > limite_superior)
            ).sum()
        )

        filas.append(
            {
                "variable": variable,
                "q1": q1,
                "q3": q3,
                "limite_inferior": limite_inferior,
                "limite_superior": limite_superior,
                "posibles_outliers": cantidad_outliers,
                "porcentaje_outliers": (
                    cantidad_outliers / len(serie) * 100
                )
            }
        )

    resultado = pd.DataFrame(filas)

    if not resultado.empty:
        resultado = resultado.sort_values(
            by="porcentaje_outliers",
            ascending=False
        ).reset_index(drop=True)

    return resultado


def mostrar_resumen_outliers(outliers):
    """
    Muestra las variables con mayor cantidad de posibles valores atípicos.
    """

    if outliers.empty:
        print("\nNo se encontraron variables numéricas para analizar outliers.")
        return

    tabla = outliers.copy()

    for columna in [
        "q1",
        "q3",
        "limite_inferior",
        "limite_superior"
    ]:
        tabla[columna] = tabla[columna].map(
            lambda valor: formatear_decimal(valor, 2)
        )

    tabla["posibles_outliers"] = tabla["posibles_outliers"].map(
        formatear_entero
    )

    tabla["porcentaje_outliers"] = tabla["porcentaje_outliers"].map(
        formatear_porcentaje
    )

    imprimir_tabla(
        tabla[
            [
                "variable",
                "q1",
                "q3",
                "limite_inferior",
                "limite_superior",
                "posibles_outliers",
                "porcentaje_outliers"
            ]
        ],
        "Resumen de posibles valores atípicos",
        max_filas=20
    )

def construir_resumen_temporal(df):
    """
    Construye un resumen del comportamiento temporal de los préstamos.
    """

    if "fecha_prestamo" not in df.columns:
        return pd.DataFrame()

    fechas = df["fecha_prestamo"].dropna()

    if fechas.empty:
        return pd.DataFrame()

    resumen = pd.DataFrame(
        {
            "indicador": [
                "Fecha mínima",
                "Fecha máxima",
                "Años diferentes",
                "Meses diferentes",
                "Días de la semana diferentes"
            ],
            "valor": [
                fechas.min().strftime("%Y-%m-%d"),
                fechas.max().strftime("%Y-%m-%d"),
                fechas.dt.year.nunique(),
                fechas.dt.to_period("M").nunique(),
                fechas.dt.dayofweek.nunique()
            ]
        }
    )

    return resumen


def mostrar_resumen_temporal(resumen_temporal):
    """
    Muestra el resumen temporal.
    """

    if resumen_temporal.empty:
        print("\nNo hay información temporal disponible.")
        return

    imprimir_tabla(
        resumen_temporal,
        "Resumen temporal de los préstamos"
    )

# Ejecución principal

def main():
    """
    Ejecuta todo el proceso de análisis exploratorio.
    """

    print("\nIniciando análisis exploratorio de datos")

    limpiar_carpeta_salida()

    df = cargar_datos()

    print(
        f"Archivo cargado correctamente: "
        f"{formatear_entero(len(df))} registros y "
        f"{formatear_entero(len(df.columns))} columnas."
    )

    resumen = construir_resumen_inicial(df)
    calidad = construir_calidad_datos(df)
    clasificacion = construir_clasificacion_variables(df)
    target = construir_target(df)
    estadisticas = construir_estadisticas_numericas(df)
    frecuencias = construir_frecuencias_categoricas(df)
    bivariado_numerico = construir_comparacion_numerica_target(df)
    bivariado_categorico = construir_comportamiento_categoricas_target(df)

    matriz_correlacion, correlaciones = construir_correlaciones(df)
    correlaciones_target = construir_correlaciones_target(df)

    hallazgos = construir_hallazgos(
        df,
        calidad,
        target,
        correlaciones_target
    )

    outliers = construir_resumen_outliers(df)
    resumen_temporal = construir_resumen_temporal(df)

    variables_numericas = [
        variable
        for variable in df.select_dtypes(
            include=np.number
        ).columns
        if variable != TARGET
    ]

    variables_categoricas = [
        variable
        for variable in VARIABLES_CATEGORICAS
        if variable in df.columns
    ]

    # Guardar tablas en CSV

    guardar_csv(
        resumen,
        "resumen_inicial.csv"
    )

    guardar_csv(
        outliers,
        "resumen_outliers.csv"
    )

    guardar_csv(
        resumen_temporal,
        "resumen_temporal.csv"
    )

    guardar_csv(
        calidad,
        "calidad_datos.csv"
    )

    guardar_csv(
        clasificacion,
        "clasificacion_variables.csv"
    )

    guardar_csv(
        target,
        "distribucion_target.csv"
    )

    guardar_csv(
        estadisticas,
        "estadisticas_numericas.csv"
    )

    guardar_csv(
        frecuencias,
        "frecuencias_categoricas.csv"
    )

    guardar_csv(
        bivariado_numerico,
        "comparacion_numerica_target.csv"
    )

    guardar_csv(
        bivariado_categorico,
        "comportamiento_categoricas_target.csv"
    )

    guardar_csv(
        correlaciones,
        "correlaciones.csv"
    )

    guardar_csv(
        correlaciones_target,
        "correlaciones_target.csv"
    )

    guardar_csv(
        hallazgos,
        "hallazgos.csv"
    )

    matriz_correlacion.to_csv(
        CARPETA_SALIDA / "matriz_correlacion.csv",
        encoding="utf-8-sig"
    )

    # Guardar tablas en Excel

    tablas_excel = {
        "Resumen": resumen,
        "Calidad_datos": calidad,
        "Clasificacion": clasificacion,
        "Target": target,
        "Estadisticas": estadisticas,
        "Frecuencias": frecuencias,
        "Bivariado_num": bivariado_numerico,
        "Bivariado_cat": bivariado_categorico,
        "Correlaciones": correlaciones,
        "Corr_target": correlaciones_target,
        "Outliers": outliers,
        "Temporal": resumen_temporal,
        "Hallazgos": hallazgos
    }

    guardar_excel(tablas_excel)

    # Crear gráficos

    crear_grafico_distribuciones_numericas(
        df,
        variables_numericas
    )

    crear_grafico_boxplots(
        df,
        variables_numericas
    )

    crear_grafico_target(df)

    crear_grafico_frecuencias_categoricas(
        df,
        variables_categoricas
    )

    crear_grafico_numericas_por_target(
        df,
        variables_numericas
    )

    crear_grafico_categoricas_por_target(
        df,
        variables_categoricas
    )

    crear_grafico_correlacion(
        matriz_correlacion
    )

    crear_grafico_relaciones_financieras(df)

    crear_grafico_puntajes(df)

    crear_grafico_temporal(df)

   # Preparar el resumen para el reporte de texto

    resumen_txt = resumen.copy()

    resumen_txt["valor"] = resumen_txt.apply(
        lambda fila: (
            formatear_porcentaje(fila["valor"])
            if fila["indicador"] == "Porcentaje total de nulos"
            else formatear_entero(fila["valor"])
        ),
        axis=1
    )

    calidad_txt = calidad.copy()

    calidad_txt["registros"] = calidad_txt["registros"].apply(
        formatear_entero
    )

    calidad_txt["valores_nulos"] = calidad_txt["valores_nulos"].apply(
        formatear_entero
    )

    calidad_txt["porcentaje_nulos"] = calidad_txt["porcentaje_nulos"].apply(
        formatear_porcentaje
    )

    calidad_txt["valores_unicos"] = calidad_txt["valores_unicos"].apply(
        formatear_entero
    )

    calidad_txt["porcentaje_unicos"] = calidad_txt["porcentaje_unicos"].apply(
        formatear_porcentaje
    )

    target_txt = target.copy()

    if not target_txt.empty:
        target_txt["cantidad"] = target_txt["cantidad"].apply(
            formatear_entero
        )

        target_txt["porcentaje"] = target_txt["porcentaje"].apply(
            formatear_porcentaje
        )

    correlaciones_target_txt = correlaciones_target.copy()

    if not correlaciones_target_txt.empty:
        correlaciones_target_txt[
            "correlacion_con_target"
        ] = correlaciones_target_txt[
            "correlacion_con_target"
        ].apply(
            lambda valor: formatear_decimal(valor, 4)
        )

        correlaciones_target_txt[
            "correlacion_absoluta"
        ] = correlaciones_target_txt[
            "correlacion_absoluta"
        ].apply(
            lambda valor: formatear_decimal(valor, 4)
        )

    hallazgos_txt = "\n".join(
        [
            f"{fila['numero']}. {fila['hallazgo']}"
            for _, fila in hallazgos.iterrows()
        ]
    )

    secciones = [
        (
            "REPORTE DE ANÁLISIS EXPLORATORIO DE DATOS",
            "Proyecto de análisis de créditos"
        ),
        (
            "1. RESUMEN GENERAL",
            tabulate(
                resumen_txt,
                headers="keys",
                tablefmt="fancy_grid",
                showindex=False
            )
        ),
        (
            "2. CALIDAD DE LOS DATOS",
            tabulate(
                calidad_txt.head(20),
                headers="keys",
                tablefmt="fancy_grid",
                showindex=False
            )
        ),
        (
            "3. DISTRIBUCIÓN DE LA VARIABLE OBJETIVO",
            tabulate(
                target_txt,
                headers="keys",
                tablefmt="fancy_grid",
                showindex=False
            )
        ),
        (
            "4. CORRELACIONES CON LA VARIABLE OBJETIVO",
            tabulate(
                correlaciones_target_txt.head(15),
                headers="keys",
                tablefmt="fancy_grid",
                showindex=False
            )
        ),
        (
            "5. HALLAZGOS PRINCIPALES",
            hallazgos_txt
        ),
        (
            "6. ARCHIVOS GENERADOS",
            "Se generaron tablas en formato CSV, un reporte consolidado "
            "en Excel, un reporte de texto y gráficos dentro de la carpeta "
            "'graficos'."
        ),
        (
            "7. CONSIDERACIONES METODOLÓGICAS",
            "El análisis exploratorio no elimina automáticamente valores "
            "atípicos ni realiza imputaciones definitivas. Estas decisiones "
            "deben establecerse posteriormente dentro del pipeline de "
            "preprocesamiento y modelado."
        )
    ]

    guardar_reporte_txt(secciones)

    # Mostrar resultados principales en consola

    imprimir_tabla(
        resumen_txt,
        "Resumen general"
    )

    imprimir_tabla(
        calidad_txt[
            [
                "variable",
                "tipo_dato",
                "valores_nulos",
                "porcentaje_nulos"
            ]
        ],
        "Calidad de los datos",
        max_filas=20
    )

    imprimir_tabla(
        clasificacion[
            [
                "variable",
                "tipo_dato",
                "clasificacion",
                "valores_nulos",
                "valores_unicos"
            ]
        ],
        "Clasificación de variables",
        max_filas=30
    )

    imprimir_tabla(
        target_txt,
        "Distribución de la variable objetivo"
    )

    mostrar_estadisticas_numericas(
        estadisticas
    )

    mostrar_frecuencias_categoricas(
        frecuencias
    )

    mostrar_comparacion_numerica_target(
        bivariado_numerico
    )

    mostrar_comportamiento_categoricas_target(
        bivariado_categorico
    )

    imprimir_tabla(
        correlaciones_target_txt[
            [
                "variable",
                "correlacion_con_target"
            ]
        ],
        "Correlaciones con Pago_atiempo",
        max_filas=20
    )

    mostrar_correlaciones(
        correlaciones
    )

    mostrar_resumen_outliers(
        outliers
    )

    mostrar_resumen_temporal(
        resumen_temporal
    )

    print("\nHallazgos principales")

    for _, fila in hallazgos.iterrows():
        print(
            f"{formatear_entero(fila['numero'])}. "
            f"{fila['hallazgo']}"
        )

    cantidad_graficos = len(
        list(CARPETA_GRAFICOS.glob("*.png"))
    )

    print("\nProceso finalizado correctamente")
    print(f"Reporte Excel: {ARCHIVO_EXCEL}")
    print(f"Reporte TXT: {ARCHIVO_TXT}")
    print(f"Carpeta de gráficos: {CARPETA_GRAFICOS}")
    print(
        f"Gráficos generados: "
        f"{formatear_entero(cantidad_graficos)}"
    )

    if cantidad_graficos == 10:
        print("Validación: se generaron correctamente los 10 gráficos.")
    else:
        print(
            "Advertencia: la cantidad de gráficos generados "
            "no es igual a 10."
        )


if __name__ == "__main__":
    main()