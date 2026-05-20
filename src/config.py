# Libraries
from pathlib import Path

# MlFlow issues
import os
os.environ["MLFLOW_TRACKING_URI"] = "http://mlflow.vanotole-lab.com"
os.environ["MLFLOW_ENABLE_PROXY_MULTIPART_UPLOAD"] = "false"

# Paths
def train_data_path() -> Path:
    """
    Returns the location of train data directory, allowing for script executions in subfolders without worrying about the
    relative location of the data

    :return: the path to the train data directory
    """
    cwd = Path.cwd()
    for folder in (cwd, cwd / "..", cwd / ".." / ".."):
        data_folder = folder / "data" / "nhanes_processed.csv"
        if data_folder.exists() and data_folder.is_file():
            print("Train data directory found in ", data_folder)
            return data_folder
    raise Exception("Train data not found")

# Cycles
CYCLES = [
"2007-2008",
"2009-2010",
"2011-2012",
"2013-2014",
"2015-2016"
]

YEARS = [
    2007,
    2009,
    2011,
    2013,
    2015
]

# DEMO - Demographics
COLS_DEMO = [
"seqn",
"ridageyr", # Edad
"riagendr", # Sexo
"ridreth1", # Raza/Etnia
"indfmpir"  # Poverty Income Ratio (PIR)
]

# BMX - Body Measures - Examination
COLS_BMX = [
"seqn",
"bmxwt",     # peso
"bmxht",     # talla
"bmxwaist",  # circunferencia de cintura
"bmxleg",    # longitud del muslo
"bmxarml",   # longitud del brazo superior
"bmxarmc"    # circunferencia del brazo superior
]

# BPX - Blood Pressure - Examination
COLS_BPX = [
"seqn",
"bpxsy1", # presión sistólica
"bpxdi1"  # presión diastólica
]

# TRIGLY - Cholesterol - Laboratory
COLS_TRIGLY = [
"seqn",
"lbxtr"   # triglicéridos
]

# HDL - Cholesterol - Laboratory
COLS_HDL = [
"seqn",
"lbdhdd"  # hdl-c
]

# GLU - Plasma Fasting Glucose & Insulin - Laboratory
COLS_GLU = [
"seqn",
"lbdglusi"  # glucosa en ayunas
]

# DIQ - Diabetes - Questionnaire
COLS_DIQ = [
"seqn",
"diq170"   # historial familiar de diabetes
]

# PAQ - Physical Activity - Questionnaire
COLS_PAQ = [
"seqn",
"pad680"   # tiempo sedentario
]

# DBQ - Diet Behavior & Nutrition - Questionnaire
COLS_DBQ = [
"seqn",
"dbd895", # comportamiento alimentario
"dbd900", # comportamiento alimentario
"dbd905", # comportamiento alimentario
"dbd910", # comportamiento alimentario
"dbq370", # comidas escolares
"dbd381", # comidas escolares
"dbq400", # comidas escolares
"dbd411", # comidas escolares
"dbq421", # comidas escolares
"dbq424", # comidas escolares
]

# Features

COL_RIDAGEYR = "ridageyr"
COL_RIAGENDR = "riagendr"
COL_RIDRETH1 = "ridreth1"
COL_INDFMPIR = "indfmpir"
COL_BMXWT = "bmxwt"
COL_BMXHT = "bmxht"
COL_BMXWAIST = "bmxwaist"
COL_BMXLEG = "bmxleg"
COL_BMXARML = "bmxarml"
COL_BMXARMC = "bmxarmc"
COL_BPXSY1 = "bpxsy1"
COL_BPXDI1 = "bpxdi1"
COL_LBXTR = "lbxtr"
COL_LBDHDD = "lbdhdd"
COL_LBDGLUSI = "lbdglusi"
COL_DIQ170 = "diq170"
COL_PAD680 = "pad680"
COL_DBD895 = "dbd895"
COL_DBD900 = "dbd900"
COL_DBD905 = "dbd905"
COL_DBD910 = "dbd910"
COL_DBQ370 = "dbq370"
COL_DBD381 = "dbd381"
COL_DBQ400 = "dbq400"
COL_DBD411 = "dbd411"
COL_DBQ421 = "dbq421"
COL_DBQ424 = "dbq424"
COL_BMI = "bmi"
COL_BMI_PERC = "bmi_perc"
TARGET = "target"

# Selección de variables para el modelo
# Se excluyen las variables diagnósticas (bmxwaist, bpxsy1, bpxdi1, lbxtr, lbdhdd, lbdglusi)
# ya que forman parte de la definición del target (MetS)

FEATURES = [
    # Demográficas
    COL_RIDAGEYR, COL_RIAGENDR, COL_RIDRETH1, COL_INDFMPIR, COL_DIQ170, COL_PAD680,
    # Medidas corporales
    COL_BMXWT, COL_BMXHT, COL_BMXLEG, COL_BMXARML, COL_BMXARMC, COL_BMI_PERC,
    # Comportamiento alimentario
    COL_DBD895, COL_DBD900, COL_DBD905, COL_DBD910,
    COL_DBQ370, COL_DBD381, COL_DBQ400, COL_DBD411, COL_DBQ421, COL_DBQ424,
]

# Top 10 features seleccionadas por importancia de permutación (Random Forest)
# Coinciden en su mayoría con las 5 seleccionadas por LASSO en Zhang et al. (2025)
#
# | Variable   | Descripción                                              |
# |------------|----------------------------------------------------------|
# | bmxwt      | Peso corporal (kg)                                       |
# | bmi_perc   | Percentil de IMC ajustado por edad y sexo (CDC 2000)     |
# | bmxleg     | Longitud del muslo (cm)                                  |
# | bmxht      | Talla (cm)                                               |
# | bmxarmc    | Circunferencia del brazo superior (cm)                   |
# | ridreth1   | Raza/etnia                                               |
# | dbd895     | Nº de comidas fuera de casa en los últimos 7 días        |
# | bmxarml    | Longitud del brazo superior (cm)                         |
# | dbd411     | Nº de veces/semana que toma desayuno escolar             |
# | dbq421     | Precio del desayuno escolar (gratis/reducido/completo)   |

FEATURES_SELECTED = [
    COL_BMXWT, COL_BMI_PERC, COL_BMXLEG, COL_BMXHT, COL_BMXARMC,
    COL_RIDRETH1, COL_DBD895, COL_BMXARML, COL_DBD411, COL_DBQ421,
]

TARGET = "target"

# Numéricas: escalar con StandardScaler
NUM_COLS = [
    COL_BMXWT, COL_BMI_PERC, COL_BMXLEG, COL_BMXHT, COL_BMXARMC,
    COL_DBD895, COL_BMXARML, COL_DBD411,
]

CAT_COLS = [
    COL_RIDRETH1, COL_DBQ421,
]

# MLFlow
MLFLOW_TRACKING_URL = "http://mlflow.vanotole-lab.com"
MLFLOW_EXPERIMENT_NAME = "Nhanes_metabolic"