import marimo

__generated_with = "0.23.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import sys
    import pandas as pd
    from scipy import stats
    import numpy as np
    np.set_printoptions(threshold=sys.maxsize)
    import os
    import matplotlib.pyplot as plt

    import miceforest as mf

    return mf, mo, np, os, pd, stats


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data import
    """)
    return


@app.cell
def _(mo, os, pd):
    # Raw NHANES data
    data_path = "../data"
    file_path = os.path.join(data_path, "nhanes_raw.csv")

    df_nhanes = pd.read_csv(filepath_or_buffer=file_path)
    df_nhanes = df_nhanes.drop(columns=['seqn'])

    # Create bmi column
    df_nhanes["bmi"] = df_nhanes["bmxwt"] / (df_nhanes["bmxht"] / 100)**2.0

    # BMI percentile (CDC 2000 paper)


    mo.ui.table(df_nhanes.head(5))
    return data_path, df_nhanes


@app.cell
def _(data_path, mo, os, pd):
    # BMI CDC 2000 percentiles parameters
    cdc_bmi_path = os.path.join(data_path,"bmiagerev.csv")

    df_cdc_bmi = pd.read_csv(cdc_bmi_path) 

    mo.ui.table(df_cdc_bmi.head())
    return (df_cdc_bmi,)


@app.cell
def _(df_nhanes):
    # NHANES DataFrame columns
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


@app.cell
def _(df_cdc_bmi, np, pd, stats):
    # Hacer interpolacion no es necesario pues la edad siempre está puesta en años enteros
    # Además, podemos utilizar los datos CDC 2000 para edades entre 2 y 20 años, pues nos estamos centrando en adolescentes solamente

    def bmi_percentile(bmi, age_years, sex):

        if pd.isna(bmi) or pd.isna(age_years) or pd.isna(sex):
            return np.nan

        df_t = df_cdc_bmi[(df_cdc_bmi["Agemos"]==(age_years*12 + 0.5)) & (df_cdc_bmi["Sex"]==sex)]
        L = df_t["L"].iloc[0]
        M = df_t["M"].iloc[0]
        S = df_t["S"].iloc[0]

        z = ((bmi/M)**L - 1) / (L*S)
        return stats.norm.cdf(z) * 100

    return (bmi_percentile,)


@app.cell
def _(bmi_percentile, df_filtered):
    df_filtered["bmi_perc"] = df_filtered.apply(
        lambda row: bmi_percentile(row['bmi'], row['ridageyr'], 1 if row['riagendr'] == "Male" else 2), axis=1
    )
    return


@app.cell
def _(df_filtered):
    df_filtered
    return


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
    kernel = mf.ImputationKernel(df_filtered.reset_index(drop=True), random_state=42)

    # Ejecutar 2 iteraciones
    kernel.mice(2, min_data_in_leaf=20) # some variables have very rare labels

    # Extraer el dataset imputado
    df_imputed = kernel.complete_data(0)

    print(f"Valores faltantes después de la imputacíon")
    print(df_imputed.isna().sum().sum())
    print(df_imputed.shape)
    return (df_imputed,)


@app.cell
def _(df_imputed):
    # Target
    # Se tienen que cumplir 3 de 5 criterios

    # Criterio 1 - cintura (depende del sexo)
    c1 = ((df_imputed["riagendr"] == "Male") & (df_imputed["bmxwaist"] >= 102)) | \
         ((df_imputed["riagendr"] == "Female") & (df_imputed["bmxwaist"] >= 88))

    # Criterio II: triglicéridos
    c2 = df_imputed["lbxtr"] >= 130

    # Criterio III: HDL-C (depende del sexo)
    c3 = ((df_imputed["riagendr"] == "Male") & (df_imputed["lbdhdd"] < 40)) | \
        ((df_imputed["riagendr"] == "Female") & (df_imputed["lbdhdd"] < 50))

    # Criterio IV: presión arterial
    c4 = (df_imputed["bpxsy1"] >= 130) | (df_imputed["bpxdi1"] >= 85)

    # Criterio V: glucosa en ayunas
    c5 = df_imputed["lbdglusi"] >= 5.6

    # Target: 1 si cumple 3 o más criterios
    df_imputed["target"] = (c1.astype(int) + c2.astype(int) + c3.astype(int) + 
                             c4.astype(int) + c5.astype(int)) >= 3
    df_imputed["target"] = df_imputed["target"].astype(int)

    print(df_imputed["target"].value_counts()) 
    return


@app.cell
def _(df_imputed):
    ## Save filtered dataset
    df_imputed.info()
    return


@app.cell
def _(df_imputed):
    df_imputed
    return


@app.cell
def _():
    #df_imputed.to_csv(path_or_buf="../data/nhanes_processed.csv", header=True, index=None)
    return


@app.cell
def _(df_imputed):
    for v in df_imputed.columns:
        print(v)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
