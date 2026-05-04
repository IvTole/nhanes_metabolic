import os
import pandas as pd
import rpy2.robjects as ro

# =========================================================
# Funciones para carga de datos (de librerías nhanes de R)
# =========================================================
# Los datos se extraen con código R y luego se guardan en un archivo csv
# El archivo csv generado se carga con pandas (opcional: polars)

def load_nhanes_data(catalog:str, cycles:list=None, years:list=None) -> pd.DataFrame:
    """
    Load NHANES data using the R package `nhanesdata` via rpy2.

    This function acts as a bridge between Python and R to retrieve NHANES datasets.
    It first checks if a local CSV file already exists. If so, it loads the data
    using pandas. Otherwise, it executes R code to download the dataset using
    `read_nhanes()`, saves it as a CSV file, and then loads it into a pandas DataFrame.

    Parameters
    ----------
    catalog : str
        Name of the NHANES dataset to load (e.g., "demo", "bpx", "bmx", "smq").

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame containing the requested NHANES dataset.

    Raises
    ------
    RuntimeError
        If there is an error during data retrieval, R execution, or file loading.

    Notes
    -----
    - Requires R to be installed and accessible from the system.
    - Requires the R package `nhanesdata` to be installed.
    - The resulting CSV file is saved in the current working directory.
    - Uses `low_memory=False` to avoid dtype inference warnings in pandas.

    Example
    -------
    df_demo = load_nhanes_data("demo")
    df_bpx = load_nhanes_data("bpx")
    """

    if not isinstance(catalog, str) or not catalog.strip():
        raise ValueError("catalog debe ser una cadena no vacía")

    catalog = catalog.strip().lower()
    file_path = "../data/"+catalog+".csv"

    try:
        if os.path.exists(file_path):
                
            print(f"Archivo {file_path} ya existe. Cargando desde CSV ...")
                
            df = pd.read_csv(filepath_or_buffer=file_path,
                             low_memory=False) # low_memory, se ignora un warning al cargar datos de diferente dtype
            
        else:
                
            print(f"Archivo {file_path} no existe. Ejecutando R para generarlo ...")

            ro.r('library(nhanesdata)')
            ro.r(f'df_all <- read_nhanes("{catalog}")')
            ro.r(f'write.csv(df_all, "{file_path}", row.names=FALSE)') # exportar datos a csv

            df = pd.read_csv(filepath_or_buffer=file_path,
                             low_memory=False)

        if cycles is not None:

            # Reescribir la columna sddsrvyr a formato YYYY-YYYY
            df["sddsrvyr"] = (
                df["sddsrvyr"]
                .str.extract(r"(\d{4}).*?(\d{4})")   # captura primer y último año
                .agg("-".join, axis=1)              # une como 'YYYY-YYYY'
            )

            df = df[df['sddsrvyr'].isin(values=cycles)].copy()

        if years is not None:

            df = df[df["year"].isin(values=years)].copy()


        print(f"{file_path} file shape: {df.shape}")

        return df

    except Exception as e:
        raise RuntimeError(
        f"Ocurrió un error al cargar el dataset NHANES '{catalog}'"
                          ) from e
