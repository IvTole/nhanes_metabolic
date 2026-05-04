# Variables

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
