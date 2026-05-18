# Predicción de Síndrome Metabólico en Adolescentes con Machine Learning (NHANES 2007–2016)

Reproducción y extensión del análisis descrito en Zhang et al. (2025), que desarrolla un modelo predictivo de síndrome metabólico (MetS) en adolescentes utilizando indicadores antropométricos y demográficos — sin marcadores bioquímicos — a partir de datos del National Health and Nutrition Examination Survey (NHANES).

## Referencia

Zhang, Y., Wu, H., Ma, R., Feng, B., Yang, R., Chen, X., Li, M., & Cheng, L. (2025). Machine learning-based predictive model for adolescent metabolic syndrome: Utilizing data from NHANES 2007–2016. *Scientific Reports*, *15*, 3274. https://doi.org/10.1038/s41598-025-88156-4

---

## Estructura del repositorio

```
nhanes_metabolic/
├── data/               # Datos descargados de NHANES y dataset final (ignorados por git)
├── notebooks/          # Análisis exploratorio en Marimo
│   └── nhanes_eda.py
├── plots/              # Gráficas generadas durante el análisis
├── train.py            # Script principal de entrenamiento y evaluación
└── src/
    ├── config.py           # Ciclos, columnas por catálogo, features y target del modelo
    ├── preprocessing.py    # ColumnTransformer: StandardScaler + OrdinalEncoder
    ├── evaluation.py       # ModelEvaluation: CV estratificada y evaluación con logging a MLFlow
    ├── plots.py            # Importancia de features por permutación
    ├── io.py               # Carga del dataset procesado (Dataset)
    ├── tracking.py         # Decorador mlflow_logger
    ├── nhanes_r_loader.py  # Descarga de catálogos NHANES desde R vía rpy2
    └── create_df.py        # Construcción y consolidación del dataset
```

---

## Orden de ejecución

```bash
# 1. Descargar catálogos NHANES y construir el dataset crudo
python src/create_df.py        # genera data/nhanes_raw.csv

# 2. Análisis exploratorio, imputación y construcción del dataset procesado
#    Abrir notebooks/nhanes_eda.py con Marimo
marimo edit notebooks/nhanes_eda.py   # genera data/nhanes_processed.csv

# 3. Entrenamiento y evaluación de modelos
python train.py
```

---

## Pasos del análisis

### 1. Descarga de datos

Se descargan 5 ciclos de NHANES (2007–2016) usando el paquete `nhanesdata` de R a través de `rpy2`. Los catálogos descargados son: DEMO, BMX, BPX, TRIGLY, HDL, GLU, DIQ, PAQ y DBQ. Se consolidan en `data/nhanes_raw.csv`.

### 2. Filtrado de participantes

- Se filtran adolescentes de 12 a 19 años.
- Se eliminan filas con valores faltantes en las variables diagnósticas: `bmxwaist`, `lbxtr`, `lbdhdd`, `lbdglusi`, `bpxsy1`, `bpxdi1`. Esto da 2,459 participantes, consistente con el paper.

### 3. Cálculo de BMI y percentil BMI

- `bmi` se calcula como `bmxwt / (bmxht / 100)²`.
- `bmi_perc` se calcula usando las tablas de crecimiento CDC 2000 con la transformación Box-Cox (parámetros L, M, S por edad en meses y sexo). Es la feature más importante del modelo.

### 4. Imputación por PMM (MICE)

Las variables predictoras tienen valores faltantes después del filtro. Se usa `miceforest` (LightGBM como regresor interno) con 2 iteraciones de MICE. Las variables diagnósticas no se imputan — ya están completas por diseño del filtro.

Parámetros relevantes:
- `min_data_in_leaf=20` para evitar probabilidades 0.0 en categorías raras (`dbq421`, `dbq424`), que causarían `inf` en el odds ratio interno y romperían el KDTree del matching.
- `reset_index(drop=True)` antes de pasar el DataFrame al kernel.

### 5. Construcción del target

MetS se define como cumplir 3 o más de los siguientes 5 criterios (AHA/NHLBI):

| Criterio | Variable | Umbral |
|----------|----------|--------|
| I. Obesidad abdominal | `bmxwaist` | ≥ 102 cm (hombres), ≥ 88 cm (mujeres) |
| II. Triglicéridos | `lbxtr` | ≥ 130 mg/dL |
| III. HDL-C | `lbdhdd` | < 40 (hombres), < 50 (mujeres) |
| IV. Presión arterial | `bpxsy1` / `bpxdi1` | ≥ 130 mmHg sistólica o ≥ 85 mmHg diastólica |
| V. Glucosa en ayunas | `lbdglusi` | ≥ 5.6 mmol/L |

Prevalencia resultante: ~5.6% (139/2,459). Las variables diagnósticas se excluyen del modelo para evitar data leakage.

### 6. Selección de features

Se excluyen las variables diagnósticas. Las features candidatas son:

```python
# Demográficas
COL_RIDAGEYR, COL_RIAGENDR, COL_RIDRETH1, COL_INDFMPIR, COL_DIQ170, COL_PAD680

# Medidas corporales
COL_BMXWT, COL_BMXHT, COL_BMXLEG, COL_BMXARML, COL_BMXARMC, COL_BMI_PERC

# Comportamiento alimentario
COL_DBD895, COL_DBD900, COL_DBD905, COL_DBD910,
COL_DBQ370, COL_DBD381, COL_DBQ400, COL_DBD411, COL_DBQ421, COL_DBQ424
```

El paper usa LASSO + 20-fold CV y selecciona 5 features: `bmi_perc`, `bmxwt`, `bmxarmc`, `bmxleg`, `ridreth1`. En este proyecto se usa importancia por permutación sobre Random Forest como alternativa (ver sección siguiente).

### 7. Importancia de features por permutación

Se entrena un Random Forest y se calcula la importancia por permutación (`sklearn.inspection.permutation_importance`) sobre el conjunto de validación. A diferencia de la importancia por impureza (Gini), la permutación mide cuánto cae el ROC-AUC al aleatorizar cada feature — más confiable cuando hay features correlacionadas.

Top 10 features identificadas (en orden descendente): `bmxwt`, `bmi_perc`, `bmxleg`, `bmxht`, `bmxarmc`, `ridreth1`, `dbd895`, `bmxarml`, `dbd411`, `dbq421`. Las primeras 5 coinciden con las seleccionadas por LASSO en el paper.

La gráfica se guarda en `plots/feature_importance.png`.

### 8. Preprocesamiento

`src/preprocessing.py` construye un `ColumnTransformer`:
- `StandardScaler` para variables numéricas.
- `OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)` para variables categóricas. El parámetro `handle_unknown` es necesario porque categorías raras como `"Don't know"` pueden aparecer solo en validación durante CV.

### 9. Entrenamiento y evaluación

Pipeline: `ColumnTransformer → SMOTE → modelo`. Se usa `imblearn.Pipeline` (no `sklearn.Pipeline`) para que SMOTE solo se aplique en los folds de training durante CV, nunca en validación.

SMOTE es necesario por el desbalance de clases (~5.6% positivos). Sin él, los modelos predicen casi todo como negativo y obtienen accuracy alto pero recall ~0.

Métricas reportadas: `accuracy`, `roc_auc`, `f1`, `precision`, `recall`, `average_precision`.

Evaluación en dos pasos:
1. `evaluate_cv()` — validación cruzada estratificada (5-fold), sin logging a MLFlow.
2. `evaluate_model()` — split fijo 80/20, con logging completo a MLFlow.

---

## Requisito: instalación del paquete `nhanesdata` en R

Este proyecto usa `rpy2` como puente entre Python y R para descargar los catálogos de NHANES.

1. Activa tu entorno:

```bash
conda activate nhanes_env
```

2. Abre R e instala el paquete:

```r
install.packages("nhanesdata")
library(nhanesdata)  # verificar instalación
```
