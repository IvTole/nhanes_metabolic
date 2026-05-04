# Predicción de Síndrome Metabólico en Adolescentes con Machine Learning (NHANES 2007–2016)

Reproducción y extensión del análisis descrito en Zhang et al. (2025), que desarrolla un modelo predictivo de síndrome metabólico (MetS) en adolescentes utilizando indicadores antropométricos y demográficos — sin marcadores bioquímicos — a partir de datos del National Health and Nutrition Examination Survey (NHANES).

## Referencia

Zhang, Y., Wu, H., Ma, R., Feng, B., Yang, R., Chen, X., Li, M., & Cheng, L. (2025). Machine learning-based predictive model for adolescent metabolic syndrome: Utilizing data from NHANES 2007–2016. *Scientific Reports*, *15*, 3274. https://doi.org/10.1038/s41598-025-88156-4

---

## Requisito: instalación del paquete `nhanesdata` en R

Este proyecto utiliza la función `load_nhanes_data` para descargar datos de NHANES mediante R, usando la librería `rpy2` como puente entre Python y R. Es necesario que el paquete `nhanesdata` esté instalado en el entorno de R que utiliza `rpy2`.

### Instalación del paquete en R

1. Activa tu entorno de trabajo (por ejemplo, en conda):

```bash
conda activate nhanes_env
```

2. Abre una sesión de R:

```bash
R
```

3. Dentro de R, ejecuta:

```r
install.packages("nhanesdata")
```

4. Para verificar la instalación:

```r
library(nhanesdata)
```
