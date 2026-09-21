from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    RocCurveDisplay
)
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC


RUTA_PROYECTO = Path(__file__).resolve().parent
RUTA_DATOS = RUTA_PROYECTO / "resultados_feature_engineering"
RUTA_SALIDA = RUTA_PROYECTO / "resultados_model_training_evaluation"

RUTA_SALIDA.mkdir(exist_ok=True)


def cargar_datos():
    """Carga los datos transformados y las etiquetas."""

    X_train = pd.read_csv(
        RUTA_DATOS / "X_train_transformado.csv"
    )

    X_test = pd.read_csv(
        RUTA_DATOS / "X_test_transformado.csv"
    )

    y_train = pd.read_csv(
        RUTA_DATOS / "y_train.csv"
    ).squeeze()

    y_test = pd.read_csv(
        RUTA_DATOS / "y_test.csv"
    ).squeeze()

    return X_train, X_test, y_train, y_test


def crear_modelos():
    """Crea los modelos de clasificación."""

    svm_base = SVC(
        class_weight="balanced",
        random_state=42
    )

    modelos = {
        "Regresion_Logistica": LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=42
        ),

        "Random_Forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),

        "SVM": CalibratedClassifierCV(
            estimator=svm_base,
            method="sigmoid",
            cv=3,
            n_jobs=-1
        )
    }

    return modelos


def imprimir_distribucion_clases(y, titulo):
    """Muestra la distribución de clases."""

    distribucion = (
        y.value_counts()
        .sort_index()
        .rename_axis("Clase")
        .reset_index(name="Cantidad")
    )

    distribucion["Porcentaje"] = (
        distribucion["Cantidad"] / len(y) * 100
    )

    print(f"\n{titulo}")
    print("-" * len(titulo))
    print(f"{'Clase':<10}{'Cantidad':>12}{'Porcentaje':>15}")
    print("-" * 37)

    for _, fila in distribucion.iterrows():
        print(
            f"{int(fila['Clase']):<10}"
            f"{int(fila['Cantidad']):>12,}"
            f"{fila['Porcentaje']:>14.2f}%"
        )


def imprimir_metricas(metricas):
    """Muestra las métricas de un modelo."""

    print("\nMétricas generales")
    print("------------------")

    print(f"{'Accuracy':<28}: {metricas['accuracy']:.4f}")
    print(
        f"{'Balanced accuracy':<28}: "
        f"{metricas['balanced_accuracy']:.4f}"
    )
    print(f"{'ROC-AUC':<28}: {metricas['roc_auc']:.4f}")
    print(f"{'PR-AUC':<28}: {metricas['pr_auc']:.4f}")

    print("\nMétricas de la clase 0 - No paga a tiempo")
    print("-----------------------------------------")

    print(
        f"{'Precision clase 0':<28}: "
        f"{metricas['precision_clase_0']:.4f}"
    )
    print(
        f"{'Recall clase 0':<28}: "
        f"{metricas['recall_clase_0']:.4f}"
    )
    print(
        f"{'F1-score clase 0':<28}: "
        f"{metricas['f1_clase_0']:.4f}"
    )

    print("\nMétricas de la clase 1 - Paga a tiempo")
    print("--------------------------------------")

    print(
        f"{'Precision clase 1':<28}: "
        f"{metricas['precision_clase_1']:.4f}"
    )
    print(
        f"{'Recall clase 1':<28}: "
        f"{metricas['recall_clase_1']:.4f}"
    )
    print(
        f"{'F1-score clase 1':<28}: "
        f"{metricas['f1_clase_1']:.4f}"
    )

    print("\nMétricas macro")
    print("--------------")

    print(
        f"{'Precision macro':<28}: "
        f"{metricas['precision_macro']:.4f}"
    )
    print(
        f"{'Recall macro':<28}: "
        f"{metricas['recall_macro']:.4f}"
    )
    print(
        f"{'F1-score macro':<28}: "
        f"{metricas['f1_macro']:.4f}"
    )


def imprimir_matriz_confusion(matriz):
    """Muestra la matriz de confusión de forma tabulada."""

    print("\nMatriz de confusión")
    print("-------------------")

    print(
        f"{'Real / Predicción':<25}"
        f"{'No paga':>12}"
        f"{'Paga':>12}"
    )
    print("-" * 49)

    print(
        f"{'No paga a tiempo':<25}"
        f"{matriz[0, 0]:>12}"
        f"{matriz[0, 1]:>12}"
    )

    print(
        f"{'Paga a tiempo':<25}"
        f"{matriz[1, 0]:>12}"
        f"{matriz[1, 1]:>12}"
    )


def imprimir_reporte_clasificacion(y_test, predicciones):
    """Muestra el reporte de clasificación en formato tabulado."""

    precision = precision_score(
        y_test,
        predicciones,
        labels=[0, 1],
        average=None,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predicciones,
        labels=[0, 1],
        average=None,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predicciones,
        labels=[0, 1],
        average=None,
        zero_division=0
    )

    soporte = pd.Series(y_test).value_counts().sort_index()

    print("\nReporte de clasificación")
    print("------------------------")

    print(
        f"{'Clase':<25}"
        f"{'Precision':>12}"
        f"{'Recall':>12}"
        f"{'F1-score':>12}"
        f"{'Soporte':>12}"
    )
    print("-" * 73)

    nombres_clases = {
        0: "No paga a tiempo",
        1: "Paga a tiempo"
    }

    for i, clase in enumerate([0, 1]):
        print(
            f"{nombres_clases[clase]:<25}"
            f"{precision[i]:>12.4f}"
            f"{recall[i]:>12.4f}"
            f"{f1[i]:>12.4f}"
            f"{int(soporte.loc[clase]):>12,}"
        )


def evaluar_modelo(nombre, modelo, X_train, X_test, y_train, y_test):
    """Entrena y evalúa un modelo."""

    print(f"\n{'=' * 70}")
    print(f"ENTRENAMIENTO: {nombre}")
    print(f"{'=' * 70}")

    modelo.fit(X_train, y_train)

    predicciones = modelo.predict(X_test)
    probabilidades = modelo.predict_proba(X_test)[:, 1]

    matriz = confusion_matrix(
        y_test,
        predicciones,
        labels=[0, 1]
    )

    metricas = {
        "Modelo": nombre,
        "accuracy": accuracy_score(y_test, predicciones),
        "balanced_accuracy": balanced_accuracy_score(
            y_test,
            predicciones
        ),
        "roc_auc": roc_auc_score(
            y_test,
            probabilidades
        ),
        "pr_auc": average_precision_score(
            y_test,
            probabilidades
        ),
        "precision_clase_0": precision_score(
            y_test,
            predicciones,
            pos_label=0,
            zero_division=0
        ),
        "recall_clase_0": recall_score(
            y_test,
            predicciones,
            pos_label=0,
            zero_division=0
        ),
        "f1_clase_0": f1_score(
            y_test,
            predicciones,
            pos_label=0,
            zero_division=0
        ),
        "precision_clase_1": precision_score(
            y_test,
            predicciones,
            pos_label=1,
            zero_division=0
        ),
        "recall_clase_1": recall_score(
            y_test,
            predicciones,
            pos_label=1,
            zero_division=0
        ),
        "f1_clase_1": f1_score(
            y_test,
            predicciones,
            pos_label=1,
            zero_division=0
        ),
        "precision_macro": precision_score(
            y_test,
            predicciones,
            average="macro",
            zero_division=0
        ),
        "recall_macro": recall_score(
            y_test,
            predicciones,
            average="macro",
            zero_division=0
        ),
        "f1_macro": f1_score(
            y_test,
            predicciones,
            average="macro",
            zero_division=0
        )
    }

    imprimir_metricas(metricas)
    imprimir_matriz_confusion(matriz)
    imprimir_reporte_clasificacion(y_test, predicciones)

    print(f"\nValidación cruzada de {nombre}")
    print("-" * (24 + len(nombre)))

    puntuaciones_cv = cross_val_score(
        modelo,
        X_train,
        y_train,
        cv=5,
        scoring="f1_macro",
        n_jobs=-1
    )

    promedio_cv = puntuaciones_cv.mean()
    desviacion_cv = puntuaciones_cv.std()

    print(
        f"{'F1-score macro promedio':<30}: "
        f"{promedio_cv:.4f}"
    )
    print(
        f"{'Desviación estándar':<30}: "
        f"{desviacion_cv:.4f}"
    )

    metricas["f1_macro_cv_promedio"] = promedio_cv
    metricas["f1_macro_cv_std"] = desviacion_cv

    ruta_modelo = RUTA_SALIDA / f"modelo_{nombre}.joblib"
    joblib.dump(modelo, ruta_modelo)

    ruta_matriz = (
        RUTA_SALIDA / f"matriz_confusion_{nombre}.png"
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matriz,
        display_labels=["No paga", "Paga"]
    )

    display.plot()
    plt.title(f"Matriz de confusión - {nombre}")
    plt.tight_layout()
    plt.savefig(ruta_matriz)
    plt.close()

    return modelo, metricas, probabilidades


def imprimir_resumen_resultados(resultados_df):
    """Muestra el resumen final con columnas alineadas."""

    print("\nRESUMEN DE RESULTADOS")
    print("---------------------")

    encabezado = (
        f"{'Modelo':<25}"
        f"{'Accuracy':>10}"
        f"{'Bal.Acc.':>10}"
        f"{'Rec. C0':>10}"
        f"{'F1 C0':>10}"
        f"{'Rec. C1':>10}"
        f"{'F1 C1':>10}"
        f"{'F1 Macro':>10}"
        f"{'ROC-AUC':>10}"
        f"{'PR-AUC':>10}"
    )

    print(encabezado)
    print("-" * len(encabezado))

    for _, fila in resultados_df.iterrows():
        nombre = fila["Modelo"].replace("_", " ")

        print(
            f"{nombre:<25}"
            f"{fila['accuracy']:>10.4f}"
            f"{fila['balanced_accuracy']:>10.4f}"
            f"{fila['recall_clase_0']:>10.4f}"
            f"{fila['f1_clase_0']:>10.4f}"
            f"{fila['recall_clase_1']:>10.4f}"
            f"{fila['f1_clase_1']:>10.4f}"
            f"{fila['f1_macro']:>10.4f}"
            f"{fila['roc_auc']:>10.4f}"
            f"{fila['pr_auc']:>10.4f}"
        )


def guardar_curva_roc(modelos_entrenados, X_test, y_test):
    """Guarda la curva ROC de los modelos."""

    plt.figure(figsize=(8, 6))

    for nombre, modelo in modelos_entrenados.items():
        probabilidades = modelo.predict_proba(X_test)[:, 1]

        RocCurveDisplay.from_predictions(
            y_test,
            probabilidades,
            name=nombre
        )

    plt.title("Curvas ROC de los modelos")
    plt.tight_layout()
    plt.savefig(RUTA_SALIDA / "curvas_roc.png")
    plt.close()


def main():
    print("\nINICIO DEL ENTRENAMIENTO Y EVALUACIÓN")
    print("-------------------------------------")

    X_train, X_test, y_train, y_test = cargar_datos()

    print("\n1. CARGA DE DATOS")
    print("------------------")
    print(
        f"{'Datos de entrenamiento':<35}: "
        f"{X_train.shape}"
    )
    print(
        f"{'Datos de prueba':<35}: "
        f"{X_test.shape}"
    )
    print(
        f"{'Cantidad de características':<35}: "
        f"{X_train.shape[1]}"
    )

    imprimir_distribucion_clases(
        y_train,
        "Distribución de clases en entrenamiento"
    )

    imprimir_distribucion_clases(
        y_test,
        "Distribución de clases en prueba"
    )

    modelos = crear_modelos()

    resultados = []
    modelos_entrenados = {}

    for nombre, modelo in modelos.items():
        modelo_entrenado, metricas, _ = evaluar_modelo(
            nombre,
            modelo,
            X_train,
            X_test,
            y_train,
            y_test
        )

        resultados.append(metricas)
        modelos_entrenados[nombre] = modelo_entrenado

    resultados_df = pd.DataFrame(resultados)

    # Se prioriza detectar la clase 0, que representa incumplimientos.
    resultados_df = resultados_df.sort_values(
        by=[
            "recall_clase_0",
            "f1_clase_0",
            "balanced_accuracy"
        ],
        ascending=False
    ).reset_index(drop=True)

    imprimir_resumen_resultados(resultados_df)

    resultados_df.to_csv(
        RUTA_SALIDA / "resultados_modelos.csv",
        index=False
    )

    columnas_cv = [
        "Modelo",
        "f1_macro_cv_promedio",
        "f1_macro_cv_std"
    ]

    resultados_df[columnas_cv].to_csv(
        RUTA_SALIDA / "validacion_cruzada.csv",
        index=False
    )

    modelo_seleccionado = resultados_df.iloc[0]["Modelo"]
    modelo_final = modelos_entrenados[modelo_seleccionado]

    # Se guarda dentro de la carpeta de resultados.
    joblib.dump(
        modelo_final,
        RUTA_SALIDA / "modelo_final.joblib"
    )

    # Se guarda también en la carpeta principal para el avance 3.
    joblib.dump(
        modelo_final,
        RUTA_PROYECTO / "modelo_final.joblib"
    )

    guardar_curva_roc(
        modelos_entrenados,
        X_test,
        y_test
    )

    print("\nCRITERIO DE SELECCIÓN")
    print("---------------------")
    print("Prioridad: detectar clientes que no pagan a tiempo")
    print("Métricas principales:")
    print("1. Recall de la clase 0")
    print("2. F1-score de la clase 0")
    print("3. Balanced accuracy")

    print(
        f"\nModelo seleccionado: "
        f"{modelo_seleccionado.replace('_', ' ')}"
    )

    print("\nArchivos principales guardados:")
    print(f"- {RUTA_PROYECTO / 'modelo_final.joblib'}")
    print(f"- {RUTA_SALIDA / 'modelo_final.joblib'}")

    print("\nResultados guardados en:")
    print(RUTA_SALIDA)

    print("\nPROCESO FINALIZADO")


if __name__ == "__main__":
    main()