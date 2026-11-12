# Tiempos de entrenamiento, evaluación y latencia por flujo

Valores calculados sobre **12 ejecuciones** por par modelo/dataset. La partición es fija; cada semilla cambia solo el orden interno de los conjuntos (y las proyecciones aleatorias de LODA).

## 1. Fases offline y evaluación

| Dataset | Modelo | Selección / validación (s) | Entrenamiento final (s) | Evaluación test (s) | Flujos/s (1/media) |
|---|---|---:|---:|---:|---:|
| NF-UNSW-NB15 | SAD | 60,89 ± 0,51 | 0,19 ± 0,00 | 0,38 ± 0,00 | 195.776 |
| NF-UNSW-NB15 | LODA | 98,56 ± 1,30 | 5,73 ± 0,07 | 6,79 ± 0,07 | 10.685 |
| NF-UNSW-NB15 | OCSVM | 0,31 ± 0,00 | 0,53 ± 0,01 | 0,32 ± 0,00 | 483.454 |
| NF-UNSW-NB15-v2 | SAD | 72,64 ± 1,16 | 0,19 ± 0,01 | 0,50 ± 0,02 | 193.374 |
| NF-UNSW-NB15-v2 | LODA | 67,38 ± 0,93 | 3,21 ± 0,06 | 5,30 ± 0,07 | 17.990 |
| NF-UNSW-NB15-v2 | OCSVM | 0,41 ± 0,00 | 0,54 ± 0,01 | 0,43 ± 0,01 | 470.943 |
| NF-CSE-CIC-IDS2018 | SAD | 144,86 ± 2,25 | 0,19 ± 0,00 | 5,40 ± 0,08 | 193.381 |
| NF-CSE-CIC-IDS2018 | LODA | 812,12 ± 29,87 | 5,71 ± 0,17 | 97,03 ± 3,76 | 10.519 |
| NF-CSE-CIC-IDS2018 | OCSVM | 4,59 ± 0,07 | 0,51 ± 0,01 | 4,75 ± 0,05 | 422.415 |

En SAD y LODA la columna de selección es la búsqueda del umbral (reentrena y evalúa cada candidato en validación). En One-Class SVM es la evaluación de validación con el cuantil fijo.

## 2. Distribución de latencia de evaluación (ms)

| Dataset | Modelo | Media | Desv. entre ejec. | Desv. por flujo | Mediana | p95 | p99 | Máx |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| NF-UNSW-NB15 | SAD | 0,0051 | 0,0000 | 0,0011 | 0,0044 | 0,0063 | 0,0068 | 0,0745 |
| NF-UNSW-NB15 | LODA | 0,0936 | 0,0010 | 0,0288 | 0,0698 | 0,1270 | 0,1285 | 0,2794 |
| NF-UNSW-NB15 | OCSVM | 0,0021 | 0,0000 | 0,0017 | 0,0005 | 0,0039 | 0,0043 | 0,0320 |
| NF-UNSW-NB15-v2 | SAD | 0,0052 | 0,0002 | 0,0011 | 0,0059 | 0,0063 | 0,0068 | 0,0577 |
| NF-UNSW-NB15-v2 | LODA | 0,0556 | 0,0008 | 0,0160 | 0,0427 | 0,0734 | 0,0762 | 0,2957 |
| NF-UNSW-NB15-v2 | OCSVM | 0,0021 | 0,0000 | 0,0017 | 0,0026 | 0,0039 | 0,0041 | 0,0243 |
| NF-CSE-CIC-IDS2018 | SAD | 0,0052 | 0,0001 | 0,0023 | 0,0059 | 0,0063 | 0,0067 | 1,4575 |
| NF-CSE-CIC-IDS2018 | LODA | 0,0951 | 0,0037 | 0,0289 | 0,0921 | 0,1277 | 0,1293 | 1,5225 |
| NF-CSE-CIC-IDS2018 | OCSVM | 0,0024 | 0,0000 | 0,0016 | 0,0036 | 0,0038 | 0,0041 | 0,0619 |

## 3. Alcance de la medida

La latencia de evaluación incluye, por flujo: construcción del registro, `score_one`, comparación con el umbral (o `classify`) y la actualización online del modelo cuando el flujo no se marca como anómalo. En One-Class SVM esa actualización incluye el mantenimiento del cuantil del `QuantileFilter`.

No incluye: codificación IP y escalado (se aplican en lote antes del bucle, como en el protocolo de evaluación), parseo del CSV, exportación/recolección NetFlow ni el cálculo de métricas y curvas.

`Desv. entre ejec.` es la desviación típica de la media de cada ejecución (n = 12); `Desv. por flujo` es la media de la desviación típica dentro de cada ejecución (jitter entre flujos). Cuando la proporción de flujos que actualizan el modelo es alta, la distribución es bimodal: los flujos marcados como anómalos no ejecutan la actualización y son mucho más rápidos.

