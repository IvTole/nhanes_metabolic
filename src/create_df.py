# Import libraries
import os
import pandas as pd
from nhanes_r_loader import load_nhanes_data
from config import CYCLES, YEARS
from config import COLS_DEMO, COLS_BMX, COLS_BPX, COLS_TRIGLY, COLS_HDL, COLS_GLU, COLS_DIQ, COLS_PAQ, COLS_DBQ

print(f"Utilizando datos de los siguientes ciclos: {CYCLES}")
print(f"Utilizando datos de los siguientes años: {YEARS} \n")

# =========================
# Variables demográficas
# =========================

print(f"Cargando datos demográficos (DEMO) ...")
df_demo_all = load_nhanes_data(catalog="demo", cycles=CYCLES)
df_demo = df_demo_all[COLS_DEMO].copy()
print(f"Datos (DEMO) cargados. Columnas: {df_demo.columns.tolist()} | Shape: {df_demo.shape} \n")

# =========================
# Variables de medidas corporales
# =========================

print(f"Cargando datos de medidas corporales (BMX) ...")
df_bmx_all = load_nhanes_data(catalog="bmx", years=YEARS)
df_bmx = df_bmx_all[COLS_BMX].copy()
print(f"Datos (BMX) cargados. Columnas: {df_bmx.columns.tolist()} | Shape: {df_bmx.shape} \n")

# =========================
# Variables de presión arterial
# =========================

print(f"Cargando datos de presión arterial (BPX) ...")
df_bpx_all = load_nhanes_data(catalog="bpx", years=YEARS)
df_bpx = df_bpx_all[COLS_BPX].copy()
print(f"Datos (BPX) cargados. Columnas: {df_bpx.columns.tolist()} | Shape: {df_bpx.shape} \n")

# =========================
# Variables de colesterol
# =========================

print(f"Cargando datos de colesterol (TRIGLY) ...")
df_trigly_all = load_nhanes_data(catalog="trigly", years=YEARS)
df_trigly = df_trigly_all[COLS_TRIGLY].copy()
print(f"Datos (TRIGLY) cargados. Columnas: {df_trigly.columns.tolist()} | Shape: {df_trigly.shape} \n")

# =========================
# Variables de colesterol (HDL)
# =========================

print(f"Cargando datos de colesterol (HDL) ...")
df_hdl_all = load_nhanes_data(catalog="hdl", years=YEARS)
df_hdl = df_hdl_all[COLS_HDL].copy()
print(f"Datos (HDL) cargados. Columnas: {df_hdl.columns.tolist()} | Shape: {df_hdl.shape} \n")

# =========================
# Variables de glucosa
# =========================

print(f"Cargando datos de glucosa (GLU) ...")
df_glu_all = load_nhanes_data(catalog="glu", years=YEARS)
df_glu = df_glu_all[COLS_GLU].copy()
print(f"Datos (GLU) cargados. Columnas: {df_glu.columns.tolist()} | Shape: {df_glu.shape} \n")

# =========================
# Variables de diabetes
# =========================

print(f"Cargando datos de diabetes (DIQ) ...")
df_diq_all = load_nhanes_data(catalog="diq", years=YEARS)
df_diq = df_diq_all[COLS_DIQ].copy()
print(f"Datos (DIQ) cargados. Columnas: {df_diq.columns.tolist()} | Shape: {df_diq.shape} \n")

# =========================
# Variables de actividad física
# =========================

print(f"Cargando datos de actividad fisica (PAQ) ...")
df_paq_all = load_nhanes_data(catalog="paq", years=YEARS)
df_paq = df_paq_all[COLS_PAQ].copy()
print(f"Datos (PAQ) cargados. Columnas: {df_paq.columns.tolist()} | Shape: {df_paq.shape} \n")

# =========================
# Variables de dieta
# =========================

print(f"Cargando datos de dieta y nutricion (DBQ) ...")
df_dbq_all = load_nhanes_data(catalog="dbq", years=YEARS)
df_dbq = df_dbq_all[COLS_DBQ].copy()
print(f"Datos (DBQ) cargados. Columnas: {df_dbq.columns.tolist()} | Shape: {df_dbq.shape} \n")

# =========================
# Join final
# =========================

print("Realizando join de todos los catálogos por SEQN ...")

dfs = [df_demo, df_bmx, df_bpx, df_trigly, df_hdl, df_glu, df_diq, df_paq, df_dbq]

df_final = dfs[0]
for df in dfs[1:]:
    df_final = df_final.merge(df, on="seqn", how="left")

print(f"Join completado. Shape final: {df_final.shape}")
print(f"Columnas finales: {df_final.columns.tolist()} \n")

# =========================
# Exportar a CSV
# =========================

output_path = "../data/nhanes_raw.csv"
df_final.to_csv(output_path, index=False)
print(f"Dataset guardado en: {output_path}")
