import marimo

__generated_with = "0.23.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import sys
    import pandas as pd
    import numpy as np
    np.set_printoptions(threshold=sys.maxsize)
    import os
    import matplotlib.pyplot as plt

    import miceforest as mf

    return mf, mo, np, os, pd, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data import
    """)
    return


@app.cell
def _(mo, os, pd):
    # Complete data
    data_path = "../data"
    file_path = os.path.join(data_path, "nhanes_raw.csv")

    df_nhanes = pd.read_csv(filepath_or_buffer=file_path)
    df_nhanes = df_nhanes.drop(columns=['seqn'])

    mo.ui.table(df_nhanes.tail(5))
    return (df_nhanes,)


@app.cell
def _(df_nhanes):
    # DataFrame columns
    for col in df_nhanes.columns:
        print(col)
    return


@app.cell
def _(df_nhanes):
    # Unique values per column (reference: paper)

    vars = ["riagendr","ridreth1", "diq170", "dbq400", "dbq421", "dbq370", "dbq424"]

    for var in vars:
        print(df_nhanes[var].value_counts(normalize=True))
        print("")
    return


@app.cell
def _(df_nhanes):
    # Info (before)
    print(df_nhanes.info())

    # DType change , "object to category"
    obj_cols = df_nhanes.select_dtypes(include="object").columns.tolist()

    for obj in obj_cols:
        df_nhanes[obj] = df_nhanes[obj].astype("category")

    # Info (after)
    print(df_nhanes.info())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data filter
    """)
    return


@app.cell
def _(df_nhanes):
    # Filtro de edad
    df_filtered = df_nhanes[df_nhanes["ridageyr"].between(12,19)]

    # Eliminar filas de variables diagnósticas
    diagnostic_cols = ["bmxwaist", "lbxtr", "lbdhdd", "lbdglusi", "bpxsy1", "bpxdi1"]

    df_filtered = df_filtered.dropna(subset=diagnostic_cols)

    print(df_filtered.info())
    return (df_filtered,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Imputación por Predictive Mean Matching (PMM)

    #### Problema

    Después de filtrar adolescentes con variables diagnósticas completas, las variables predictoras (como antropométricas o cuestionarios) todavía tienen valores faltantes. Después del filtro anterior ya tenemos pocos datos, así que no conviene reducirlas innecesariamente. Utilizamos el método PMM (Predictive Mean Matching) para imputar estos valores, preservando la distribución original de los datos.

    #### Algoritmo

    Para cada variable `Y` con valores faltantes, dado un conjunto de predictores `X`:

    **1. Ajuste del modelo de regresión**

    Se ajusta una regresión lineal sobre los casos observados:

    \begin{equation}
    Y_{\rm obs} = X_{\rm obs} \beta + \epsilon
    \end{equation}

    De los parámetros $\beta$ ajustados, podemos obtener una predicción para todos los casos (observados y faltantes),

    \begin{equation}
    \hat{y} = X \cdot \beta
    \end{equation}

    **2. Matching**

    Para cada caso faltante $i$, se buscan los $k$ casos observados $j$ cuyo valor predicho es más cercano en distancia con métrica euclideana,

    \begin{equation}
    d(i,j) = |{\hat{y}_i} - \hat{y}_j|
    \end{equation}

    Se selecciona aleatoriamente uno de esos $k$ donantes.

    **3. Imputación**

    El valor imputado es el valor **real observado** del donante seleccionado, no el valor predicho,

    \begin{equation}
    y_{\rm imputed \ i} = Y_{\rm obs \ i}
    \end{equation}

    Esto garantiza que los valores imputados pretencesn al soporte real de los datos.

    **Diferencias aquí**

    Cuando hay múltiples variables con valores faltantes simultáneos, PMM se aplica de forma iterativa en un esquema llamado MICE,

    1. Inicializar todos los valores faltantes con valores aleatorios de la distribución observada
    2. Para cada variable $y_i$ con valores nulos, se imputa usando las demás como variables predictoras (incluyendo las ya imputadas en esta iteración)
    3. Repetimos este ciclo tantas iteraciones hasta la convergencia

    La idea es que cada iteración refina las imputaciones usando información más completa. Con 2 iteraciones sería suficiente para este tipo de datos tabulares de tamaño moderado.

    En este proyecto, utilizaremos la librería `miceforest` para reemplazar la regresión lineal del paso 1 con LightGBM, lo que maneja automáticamente variables categóricas, relaciones no lineales, e interacciones sin necesidad de procesamiento adicional.
    """)
    return


@app.cell
def _(df_filtered, np):
    # Columnas con inf
    inf_cols = df_filtered.select_dtypes(include="number").columns[df_filtered.select_dtypes(include="number").apply(lambda c:
      np.isinf(c).any())]
    print("Columnas con inf:", inf_cols.tolist())

    # Porcentaje de missing por columna
    missing = df_filtered.isna().mean().sort_values(ascending=False)
    print("\nMissing (%):\n", (missing[missing > 0] * 100).round(1))
    return


@app.cell
def _(df_filtered, mf):
    # PMM con miceforest

    # Normalmente separariamos el target antes de empezar (no queremos que participe)
    # Pero aquí todavía no hemos creado un target para el sindrome metabolico

    # kernel de imputacion
    kernel = mf.ImputationKernel(df_filtered.reset_index(), random_state=42)

    # Ejecutar 2 iteraciones
    kernel.mice(2)

    # Extraer el dataset imputado
    df_imputed = kernel.complete_data(0)

    print(f"Valores faltantes después de la imputacíon")
    print(df_imputed.isna().sum().sum())
    print(df_imputed.shape)
    return


@app.cell
def _(df_filtered):
    # Target
    # Se tienen que cumplir 3 de 5 criterios

    # Criterio 1 - cintura (depende del sexo)
    c1 = ((df_filtered["riagendr"] == "Male") & (df_filtered["bmxwaist"] >= 102)) | \
         ((df_filtered["riagendr"] == "Female") & (df_filtered["bmxwaist"] >= 88))

    # Criterio II: triglicéridos
    c2 = df_filtered["lbxtr"] >= 130

    # Criterio III: HDL-C (depende del sexo)
    c3 = ((df_filtered["riagendr"] == "Male") & (df_filtered["lbdhdd"] < 40)) | \
        ((df_filtered["riagendr"] == "Female") & (df_filtered["lbdhdd"] < 50))

    # Criterio IV: presión arterial
    c4 = (df_filtered["bpxsy1"] >= 130) | (df_filtered["bpxdi1"] >= 85)

    # Criterio V: glucosa en ayunas
    c5 = df_filtered["lbdglusi"] >= 5.6

    # Target: 1 si cumple 3 o más criterios
    df_filtered["target"] = (c1.astype(int) + c2.astype(int) + c3.astype(int) + 
                             c4.astype(int) + c5.astype(int)) >= 3
    df_filtered["target"] = df_filtered["target"].astype(int)

    print(df_filtered["target"].value_counts()) 
    return


@app.cell
def _(df_filtered, plt):
    # Variable definition and variable selection

    plt.hist(df_filtered['pad680'])
    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
