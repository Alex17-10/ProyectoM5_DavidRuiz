import pandas as pd
import numpy as np
from pathlib import Path
from tabulate import tabulate

ruta_base = Path(__file__).resolve().parent
ruta_csv = ruta_base / "Base_de_datos.csv"

df = pd.read_csv(ruta_csv)

filas_originales, columnas_originales = df.shape
celdas_originales = filas_originales * columnas_originales

print("\nCARGA INICIAL DE DATOS")
print(f"Filas: {filas_originales:,}")
print(f"Columnas: {columnas_originales}")
print(f"Celdas: {celdas_originales:,}")

print("\nCOLUMNAS DEL DATASET")
print(df.columns.tolist())

nulos_originales = int(df.isna().sum().sum())
duplicados_originales = int(df.duplicated().sum())

print("\nCALIDAD INICIAL")
print(f"Valores nulos: {nulos_originales:,}")
print(f"Porcentaje de nulos: {(nulos_originales / celdas_originales) * 100:.2f}%")
print(f"Filas duplicadas: {duplicados_originales:,}")

print("\nNULOS POR COLUMNA")
nulos_por_columna = df.isna().sum()
nulos_tabla = pd.DataFrame({
    "Columna": nulos_por_columna.index,
    "Valores nulos": nulos_por_columna.values,
    "Porcentaje": (nulos_por_columna.values / filas_originales * 100).round(2)
})
print(
    tabulate(
        nulos_tabla[nulos_tabla["Valores nulos"] > 0],
        headers="keys",
        tablefmt="grid",
        showindex=False
    )
)

columnas_numericas = [
    "capital_prestado",
    "plazo_meses",
    "edad_cliente",
    "salario_cliente",
    "total_otros_prestamos",
    "cuota_pactada",
    "puntaje",
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
    "Pago_atiempo"
]

for columna in columnas_numericas:
    if columna in df.columns:
        df[columna] = pd.to_numeric(df[columna], errors="coerce")

print("\nTIPOS DE DATOS INICIALES")
tipos_tabla = pd.DataFrame({
    "Columna": df.dtypes.index,
    "Tipo de dato": df.dtypes.astype(str).values
})
print(tabulate(tipos_tabla, headers="keys", tablefmt="grid", showindex=False))

print("\nLIMPIEZA DE FECHAS")

df["fecha_prestamo"] = pd.to_datetime(
    df["fecha_prestamo"],
    errors="coerce",
    format="mixed"
)

fechas_invalidas = int(df["fecha_prestamo"].isna().sum())
fechas_futuras = int(
    (df["fecha_prestamo"] > pd.Timestamp.now()).sum()
)

print(f"Fechas inválidas convertidas a nulos: {fechas_invalidas}")
print(f"Fechas futuras detectadas: {fechas_futuras}")

if fechas_futuras > 0:
    df.loc[
        df["fecha_prestamo"] > pd.Timestamp.now(),
        "fecha_prestamo"
    ] = pd.NaT

fecha_minima = df["fecha_prestamo"].min()
fecha_maxima = df["fecha_prestamo"].max()

print(f"Fecha mínima válida: {fecha_minima}")
print(f"Fecha máxima válida: {fecha_maxima}")

print("\nLIMPIEZA DE EDADES")

edades_invalidas = (
    (df["edad_cliente"] < 18) |
    (df["edad_cliente"] > 100)
).sum()

df.loc[
    (df["edad_cliente"] < 18) |
    (df["edad_cliente"] > 100),
    "edad_cliente"
] = np.nan

print(f"Edades fuera del rango 18-100: {edades_invalidas}")
print(f"Edades nulas después de la limpieza: {df['edad_cliente'].isna().sum()}")

print("\nVALIDACIÓN DE VARIABLES NUMÉRICAS")

validaciones = []

for columna in [
    "capital_prestado",
    "salario_cliente",
    "cuota_pactada",
    "total_otros_prestamos",
    "plazo_meses",
    "cant_creditosvigentes",
    "huella_consulta",
    "creditos_sectorFinanciero",
    "creditos_sectorCooperativo",
    "creditos_sectorReal"
]:
    cantidad = int((df[columna] < 0).sum())
    validaciones.append([columna, "Valores negativos", cantidad])

cantidad_plazos_invalidos = int((df["plazo_meses"] <= 0).sum())
validaciones.append([
    "plazo_meses",
    "Valores menores o iguales a cero",
    cantidad_plazos_invalidos
])

print(
    tabulate(
        pd.DataFrame(
            validaciones,
            columns=["Variable", "Validación", "Cantidad"]
        ),
        headers="keys",
        tablefmt="grid",
        showindex=False
    )
)

for columna in [
    "capital_prestado",
    "salario_cliente",
    "cuota_pactada",
    "total_otros_prestamos",
    "plazo_meses",
    "cant_creditosvigentes",
    "huella_consulta",
    "creditos_sectorFinanciero",
    "creditos_sectorCooperativo",
    "creditos_sectorReal"
]:
    df.loc[df[columna] < 0, columna] = np.nan

df.loc[df["plazo_meses"] <= 0, "plazo_meses"] = np.nan

print("\nLIMPIEZA DE PUNTAJES")

puntaje_negativo = int((df["puntaje"] < 0).sum())
datacredito_negativo = int((df["puntaje_datacredito"] < 0).sum())

df.loc[df["puntaje"] < 0, "puntaje"] = np.nan
df.loc[
    df["puntaje_datacredito"] < 0,
    "puntaje_datacredito"
] = np.nan

print(f"Valores negativos en puntaje: {puntaje_negativo}")
print(f"Valores negativos en puntaje_datacredito: {datacredito_negativo}")

print("\nVALIDACIÓN DE SALDOS")

principal_mayor_total = int(
    (
        df["saldo_principal"] > df["saldo_total"]
    ).sum()
)

mora_mayor_total = int(
    (
        df["saldo_mora"] > df["saldo_total"]
    ).sum()
)

mora_codeudor_negativa = int(
    (df["saldo_mora_codeudor"] < 0).sum()
)

print(f"Saldo principal mayor que saldo total: {principal_mayor_total}")
print(f"Saldo de mora mayor que saldo total: {mora_mayor_total}")
print(f"Saldo de mora del codeudor negativo: {mora_codeudor_negativa}")

df.loc[
    df["saldo_mora_codeudor"] < 0,
    "saldo_mora_codeudor"
] = np.nan

print("\nLIMPIEZA DE CATEGORÍAS")

valores_tendencia_originales = df["tendencia_ingresos"].value_counts(
    dropna=False
)

categorias_tendencia_validas = [
    "Creciente",
    "Decreciente",
    "Estable"
]

df["tendencia_ingresos"] = df["tendencia_ingresos"].where(
    df["tendencia_ingresos"].isin(categorias_tendencia_validas),
    np.nan
)

valores_tendencia_invalidos = int(
    (~df["tendencia_ingresos"].isin(categorias_tendencia_validas) &
     df["tendencia_ingresos"].notna()).sum()
)

print("Valores de tendencia de ingresos:")
print(
    tabulate(
        df["tendencia_ingresos"].value_counts(dropna=False).reset_index(),
        headers=["Categoría", "Cantidad"],
        tablefmt="grid",
        showindex=False
    )
)

print("\nVALIDACIÓN DE SALARIOS")

salarios_cero_o_negativos = int(
    (df["salario_cliente"] <= 0).sum()
)

df.loc[
    df["salario_cliente"] <= 0,
    "salario_cliente"
] = np.nan

print(
    f"Salarios menores o iguales a cero convertidos a nulos: "
    f"{salarios_cero_o_negativos}"
)

print("\nVALIDACIÓN DE CUOTAS FRENTE AL SALARIO")

cuota_mayor_salario = int(
    (
        (df["salario_cliente"] > 0) &
        (df["cuota_pactada"] > df["salario_cliente"])
    ).sum()
)

print(f"Registros donde la cuota supera el salario: {cuota_mayor_salario}")

print("\nDETECCIÓN DE VALORES EXTREMOS")

columnas_extremos = [
    "salario_cliente",
    "capital_prestado",
    "total_otros_prestamos",
    "cuota_pactada",
    "puntaje",
    "puntaje_datacredito"
]

extremos_resultados = []

for columna in columnas_extremos:
    serie = df[columna].dropna()

    if serie.empty:
        continue

    q1 = serie.quantile(0.25)
    q3 = serie.quantile(0.75)
    iqr = q3 - q1
    limite_inferior = q1 - 1.5 * iqr
    limite_superior = q3 + 1.5 * iqr

    if iqr == 0:
        cantidad_extremos = 0
    else:
        cantidad_extremos = int(
            (
                (serie < limite_inferior) |
                (serie > limite_superior)
            ).sum()
        )

    extremos_resultados.append([
        columna,
        round(q1, 2),
        round(q3, 2),
        round(limite_inferior, 2),
        round(limite_superior, 2),
        cantidad_extremos,
        round(serie.min(), 2),
        round(serie.max(), 2)
    ])

extremos_tabla = pd.DataFrame(
    extremos_resultados,
    columns=[
        "Variable",
        "Q1",
        "Q3",
        "Límite inferior",
        "Límite superior",
        "Extremos detectados",
        "Mínimo",
        "Máximo"
    ]
)

print(
    tabulate(
        extremos_tabla,
        headers="keys",
        tablefmt="grid",
        showindex=False
    )
)

print("\nCREACIÓN DE VARIABLES DERIVADAS")

df["anio_prestamo"] = df["fecha_prestamo"].dt.year
df["mes_prestamo"] = df["fecha_prestamo"].dt.month
df["dia_semana_prestamo"] = df["fecha_prestamo"].dt.dayofweek
df["hora_prestamo"] = df["fecha_prestamo"].dt.hour

df["relacion_cuota_salario"] = np.where(
    df["salario_cliente"] > 0,
    df["cuota_pactada"] / df["salario_cliente"],
    np.nan
)

df["relacion_deuda_salario"] = np.where(
    df["salario_cliente"] > 0,
    (
        df["total_otros_prestamos"] +
        df["capital_prestado"]
    ) / df["salario_cliente"],
    np.nan
)

df["saldo_pendiente_estimado"] = (
    df["saldo_total"] - df["saldo_principal"]
)

df.loc[
    df["saldo_pendiente_estimado"] < 0,
    "saldo_pendiente_estimado"
] = np.nan

df.replace([np.inf, -np.inf], np.nan, inplace=True)

print("\nVALIDACIÓN DE VARIABLES DERIVADAS")

cuota_salario_mayor_uno = int(
    (df["relacion_cuota_salario"] > 1).sum()
)

deuda_salario_mayor_uno = int(
    (df["relacion_deuda_salario"] > 1).sum()
)

saldo_pendiente_negativo = int(
    (df["saldo_pendiente_estimado"] < 0).sum()
)

print(
    f"Relación cuota/salario mayor a 1: "
    f"{cuota_salario_mayor_uno}"
)

print(
    f"Relación deuda/salario mayor a 1: "
    f"{deuda_salario_mayor_uno}"
)

print(
    f"Saldo pendiente estimado negativo: "
    f"{saldo_pendiente_negativo}"
)

print("\nVALIDACIÓN DEL OBJETIVO")

valores_objetivo = sorted(
    df["Pago_atiempo"].dropna().unique().tolist()
)

distribucion_objetivo = (
    df["Pago_atiempo"]
    .value_counts(dropna=False)
    .rename_axis("Valor")
    .reset_index(name="Cantidad")
)

distribucion_objetivo["Porcentaje"] = (
    distribucion_objetivo["Cantidad"] /
    len(df) *
    100
).round(2)

print(f"Valores encontrados en Pago_atiempo: {valores_objetivo}")
print(
    tabulate(
        distribucion_objetivo,
        headers="keys",
        tablefmt="grid",
        showindex=False
    )
)

valores_objetivo_invalidos = int(
    (~df["Pago_atiempo"].isin([0, 1]) &
     df["Pago_atiempo"].notna()).sum()
)

print(f"Valores inválidos en Pago_atiempo: {valores_objetivo_invalidos}")

if valores_objetivo_invalidos > 0:
    df.loc[
        ~df["Pago_atiempo"].isin([0, 1]),
        "Pago_atiempo"
    ] = np.nan

print("\nESTADÍSTICAS DESCRIPTIVAS FINALES")

estadisticas = df.describe(include="all").transpose()
estadisticas = estadisticas.reset_index()
estadisticas = estadisticas.rename(columns={"index": "Variable"})

print(
    tabulate(
        estadisticas,
        headers="keys",
        tablefmt="grid",
        showindex=False
    )
)

print("\nTIPOS DE DATOS FINALES")

tipos_finales = pd.DataFrame({
    "Columna": df.dtypes.index,
    "Tipo de dato": df.dtypes.astype(str).values
})

print(
    tabulate(
        tipos_finales,
        headers="keys",
        tablefmt="grid",
        showindex=False
    )
)

print("\nNULOS FINALES")

nulos_finales = df.isna().sum()
nulos_finales_tabla = pd.DataFrame({
    "Columna": nulos_finales.index,
    "Valores nulos": nulos_finales.values,
    "Porcentaje": (
        nulos_finales.values / len(df) * 100
    ).round(2)
})

print(
    tabulate(
        nulos_finales_tabla[nulos_finales_tabla["Valores nulos"] > 0],
        headers="keys",
        tablefmt="grid",
        showindex=False
    )
)

print("\nPRIMERAS FILAS")

primeras = df.head().copy()

for inicio in range(0, len(primeras.columns), 8):
    grupo = primeras.iloc[:, inicio:inicio + 8]
    print(
        tabulate(
            grupo,
            headers="keys",
            tablefmt="grid",
            showindex=True
        )
    )

print("\nÚLTIMAS FILAS")

ultimas = df.tail().copy()

for inicio in range(0, len(ultimas.columns), 8):
    grupo = ultimas.iloc[:, inicio:inicio + 8]
    print(
        tabulate(
            grupo,
            headers="keys",
            tablefmt="grid",
            showindex=True
        )
    )

print("\nFRECUENCIAS DE VARIABLES CATEGÓRICAS")

columnas_categoricas = [
    "tipo_credito",
    "tipo_laboral",
    "tendencia_ingresos"
]

for columna in columnas_categoricas:
    frecuencia = (
        df[columna]
        .value_counts(dropna=False)
        .rename_axis("Categoría")
        .reset_index(name="Cantidad")
    )

    frecuencia["Porcentaje"] = (
        frecuencia["Cantidad"] / len(df) * 100
    ).round(2)

    print(f"\n{columna}")
    print(
        tabulate(
            frecuencia,
            headers="keys",
            tablefmt="grid",
            showindex=False
        )
    )

print("\nRESUMEN FINAL")

nulos_finales_total = int(df.isna().sum().sum())
filas_finales, columnas_finales = df.shape
celdas_finales = filas_finales * columnas_finales
infinitos_finales = int(
    np.isinf(
        df.select_dtypes(include=np.number)
    ).sum().sum()
)

resumen_final = pd.DataFrame([
    ["Filas iniciales", filas_originales],
    ["Columnas iniciales", columnas_originales],
    ["Filas finales", filas_finales],
    ["Columnas finales", columnas_finales],
    ["Filas eliminadas", filas_originales - filas_finales],
    ["Duplicados detectados", duplicados_originales],
    ["Nulos iniciales", nulos_originales],
    ["Porcentaje de nulos inicial", round(
        nulos_originales / celdas_originales * 100, 2
    )],
    ["Nulos finales", nulos_finales_total],
    ["Porcentaje de nulos final", round(
        nulos_finales_total / celdas_finales * 100, 2
    )],
    ["Edades inválidas corregidas", int(edades_invalidas)],
    ["Puntajes negativos corregidos", puntaje_negativo],
    ["Puntajes Datacrédito negativos corregidos", datacredito_negativo],
    ["Salarios menores o iguales a cero", salarios_cero_o_negativos],
    ["Cuotas superiores al salario", cuota_mayor_salario],
    ["Relación cuota/salario mayor a 1", cuota_salario_mayor_uno],
    ["Relación deuda/salario mayor a 1", deuda_salario_mayor_uno],
    ["Valores infinitos finales", infinitos_finales],
    ["Valores inválidos del objetivo", valores_objetivo_invalidos]
], columns=["Indicador", "Valor"])

print(
    tabulate(
        resumen_final,
        headers="keys",
        tablefmt="grid",
        showindex=False
    )
)

ruta_salida = ruta_base / "Base_de_datos_preparada.csv"
ruta_observaciones = ruta_base / "Observaciones_calidad_datos.csv"

df.to_csv(ruta_salida, index=False, encoding="utf-8-sig")

observaciones = pd.DataFrame({
    "Indicador": [
        "Filas iniciales",
        "Columnas iniciales",
        "Filas finales",
        "Columnas finales",
        "Filas eliminadas",
        "Duplicados detectados",
        "Nulos iniciales",
        "Nulos finales",
        "Edades inválidas corregidas",
        "Puntajes negativos corregidos",
        "Puntajes Datacrédito negativos corregidos",
        "Salarios menores o iguales a cero",
        "Cuotas superiores al salario",
        "Relación cuota/salario mayor a 1",
        "Relación deuda/salario mayor a 1",
        "Valores infinitos finales",
        "Valores inválidos del objetivo"
    ],
    "Cantidad": [
        filas_originales,
        columnas_originales,
        filas_finales,
        columnas_finales,
        filas_originales - filas_finales,
        duplicados_originales,
        nulos_originales,
        nulos_finales_total,
        edades_invalidas,
        puntaje_negativo,
        datacredito_negativo,
        salarios_cero_o_negativos,
        cuota_mayor_salario,
        cuota_salario_mayor_uno,
        deuda_salario_mayor_uno,
        infinitos_finales,
        valores_objetivo_invalidos
    ]
})

observaciones.to_csv(
    ruta_observaciones,
    index=False,
    encoding="utf-8-sig"
)

print("\nARCHIVOS GENERADOS")
print(f"Base preparada: {ruta_salida}")
print(f"Informe de calidad: {ruta_observaciones}")
print("\nProceso finalizado correctamente.")