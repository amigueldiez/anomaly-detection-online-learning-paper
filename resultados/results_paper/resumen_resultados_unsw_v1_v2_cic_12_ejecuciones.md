# Resumen de resultados: NF-UNSW-NB15, NF-UNSW-NB15-v2 y NF-CSE-CIC-IDS2018

Todos los valores son de **test held-out, media ± desviación típica con n = 12 ejecuciones**.

## 1. Resultados tras las 12 ejecuciones

| Dataset | Modelo | Accuracy | Recall | FPR | Precision | F1 | Parámetro elegido |
|---|---|---:|---:|---:|---:|---:|---|
| **NF-UNSW-NB15** | SAD | 97,6051 ± 0,3651 % | **100,0000 ± 0,0000 %** | 4,7899 ± 0,7302 % | 95,4333 ± 0,6622 % | 97,6622 ± 0,3475 % | umbral 4,8250σ ± 0,1288σ |
| NF-UNSW-NB15 | LODA | 96,7155 ± 0,3863 % | 99,9892 ± 0,0375 % | 6,5582 ± 0,7856 % | 93,8496 ± 0,6959 % | 96,8209 ± 0,3631 % | q 0,9567 ± 0,0156 |
| NF-UNSW-NB15 | **One-Class SVM** | **97,9907 ± 0,0059 %** | **100,0000 ± 0,0000 %** | **4,0185 ± 0,0118 %** | **96,1367 ± 0,0109 %** | **98,0303 ± 0,0057 %** | q = 0,99 fijo (ν=0,05, lr=0,1) |
| **NF-UNSW-NB15-v2** | SAD | 99,8894 ± 0,0047 % | **100,0000 ± 0,0000 %** | 0,2213 ± 0,0094 % | 99,7792 ± 0,0094 % | 99,8895 ± 0,0047 % | umbral 6,2500σ ± 0,7416σ |
| NF-UNSW-NB15-v2 | LODA | 97,6758 ± 1,1332 % | 99,9485 ± 0,1786 % | 4,5969 ± 2,2031 % | 95,6409 ± 2,0149 % | 97,7378 ± 1,0837 % | q 0,9250 ± 0,0261 |
| NF-UNSW-NB15-v2 | **One-Class SVM** | **99,9637 ± 0,0403 %** | **100,0000 ± 0,0000 %** | **0,0726 ± 0,0806 %** | **99,9275 ± 0,0803 %** | **99,9637 ± 0,0402 %** | q = 0,99 fijo (ν=0,05, lr=0,3) |
| **NF-CSE-CIC-IDS2018** | **SAD** | **91,1956 ± 0,0258 %** | 89,1866 ± 0,0470 % | **6,7954 ± 0,0427 %** | **92,9201 ± 0,0403 %** | **91,0151 ± 0,0269 %** | umbral 3,0σ ± 0,0σ |
| NF-CSE-CIC-IDS2018 | LODA | 90,3442 ± 4,1900 % | **91,8797 ± 1,9985 %** | 11,1913 ± 9,5363 % | 89,7759 ± 6,5839 % | 90,6385 ± 3,2366 % | q 0,9167 ± 0,0718 |
| NF-CSE-CIC-IDS2018 | One-Class SVM | 78,0002 ± 0,0524 % | 69,7872 ± 0,1731 % | 13,7869 ± 0,1735 % | 83,5038 ± 0,1468 % | 76,0316 ± 0,0714 % | q = 0,9 fijo (ν=0,5, lr=0,3) |

En negrita, el mejor valor de cada columna dentro de cada dataset.

> **Nota (2026-11-12).** Los nueve pares modelo–dataset se han vuelto a ejecutar
> con el protocolo actual y semillas `0–11`. SAD y LODA reproducen exactamente
> los valores anteriores. One-Class SVM en UNSW se ha re-ejecutado con semillas
> fijas (antes eran 12 ejecuciones sin semilla), por lo que sus cifras se han
> actualizado; en CIC no cambia. Los artefactos están en
> `resultados/multisemilla_sad_loda_unsw.json` y
> `resultados/multisemilla_cic_ids2018.json`, y las tablas de tiempos en
> `resultados/results_paper/tabla_tiempos_12_ejecuciones.md`.

## 2. Parámetros por par modelo-dataset

| Dataset | Modelo | Scaler | Modelo / hiperparámetros | Umbral / selección | Semillas |
|---|---|---|---|---|---|
| NF-UNSW-NB15 | SAD | `MaxAbsScaler` | Un `StandardAbsoluteDeviation` por característica, `sub_stat=Mean`, `var=Var`, agregación `max`, filtro de varianza `> 1e-12` | Rejilla `0,0–10,0σ` paso `0,1` + `12/15/20` (104 candidatos); F1 en validación; empates → mediana del plateau | `0–11` |
| NF-UNSW-NB15-v2 | SAD | `MaxAbsScaler` | Ídem | Ídem | `0–11` |
| NF-CSE-CIC-IDS2018 | SAD | `MaxAbsScaler` | Ídem | Rejilla `0,0–10,0σ` paso `0,5` + `12/15/20` (24 candidatos); F1 en validación | `0–11` |
| NF-UNSW-NB15 | LODA | `Normalizer` | `LODA(n_bins=5, n_random_cuts=100, seed=s)` | Cuantiles de los scores de entrenamiento: `0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99`; F1 en validación; reentreno online con flujos no marcados | `0–11` |
| NF-UNSW-NB15-v2 | LODA | `StandardScaler` | `LODA(n_bins=20, n_random_cuts=50, seed=s)` | Ídem | `0–11` |
| NF-CSE-CIC-IDS2018 | LODA | `Normalizer` | `LODA(n_bins=5, n_random_cuts=100, seed=s)` | Ídem | `0–11` |
| NF-UNSW-NB15 | One-Class SVM | `MaxAbsScaler` | `QuantileFilter(OneClassSVM(nu=0.05, intercept_lr=InverseScaling(0.1)), q=0.99)` | Una sola configuración; F1 en validación | `0–11` |
| NF-UNSW-NB15-v2 | One-Class SVM | `MaxAbsScaler` | `QuantileFilter(OneClassSVM(nu=0.05, intercept_lr=InverseScaling(0.3)), q=0.99)` | Una sola configuración; F1 en validación | `0–11` |
| NF-CSE-CIC-IDS2018 | One-Class SVM | `StandardScaler` | `QuantileFilter(OneClassSVM(nu=0.5, intercept_lr=InverseScaling(0.3)), q=0.9)` | Configuración fija (mejor de la rejilla de 48); F1 en validación | `0–11` |

## 3. Parámetros comunes a todos los pares

| Parámetro | Valor |
|---|---|
| `FLOWS_TRAIN_SCALER` | `1000` |
| `FLOWS_TRAIN_OML` | `100000` |
| `VALIDATION_RATIO` | `0.5` |
| `SELECTION_METRIC` | `f1` |
| Features | `IPV4_SRC_ADDR, IPV4_DST_ADDR, L4_DST_PORT, L4_SRC_PORT, PROTOCOL, IN_BYTES, OUT_BYTES, FLOW_DURATION_MILLISECONDS, Label` |
| Validación / test | Balanceados al 50 % |
| Partición UNSW | Posicional sobre el CSV original (bloques consecutivos) |
| Partición CIC | CSV pre-barajado con `random.seed(42)` |

## 4. Tiempos de entrenamiento, evaluación y latencia

Todas las cifras son media ± desviación típica sobre **12 ejecuciones**. En SAD y
LODA la columna de selección es la búsqueda del umbral (reentrena y evalúa cada
candidato en validación); en One-Class SVM es la evaluación de validación con el
cuantil fijo.

| Dataset | Modelo | Selección / validación (s) | Entrenamiento final (s) | Evaluación test (s) | Flujos/s (1/media) |
|---|---|---:|---:|---:|---:|
| NF-UNSW-NB15 | SAD | 60,89 ± 0,51 | 0,19 ± 0,00 | 0,38 ± 0,00 | 195.776 |
| NF-UNSW-NB15 | LODA | 98,56 ± 1,30 | 5,73 ± 0,07 | 6,79 ± 0,07 | 10.685 |
| NF-UNSW-NB15 | One-Class SVM | 0,31 ± 0,00 | 0,53 ± 0,01 | 0,32 ± 0,00 | 483.454 |
| NF-UNSW-NB15-v2 | SAD | 72,64 ± 1,16 | 0,19 ± 0,01 | 0,50 ± 0,02 | 193.374 |
| NF-UNSW-NB15-v2 | LODA | 67,38 ± 0,93 | 3,21 ± 0,06 | 5,30 ± 0,07 | 17.990 |
| NF-UNSW-NB15-v2 | One-Class SVM | 0,41 ± 0,00 | 0,54 ± 0,01 | 0,43 ± 0,01 | 470.943 |
| NF-CSE-CIC-IDS2018 | SAD | 144,86 ± 2,25 | 0,19 ± 0,00 | 5,40 ± 0,08 | 193.381 |
| NF-CSE-CIC-IDS2018 | LODA | 812,12 ± 29,87 | 5,71 ± 0,17 | 97,03 ± 3,76 | 10.519 |
| NF-CSE-CIC-IDS2018 | One-Class SVM | 4,59 ± 0,07 | 0,51 ± 0,01 | 4,75 ± 0,05 | 422.415 |

Distribución de la latencia de evaluación por flujo (ms). `Desv. entre ejec.` es
la desviación típica de la media de cada ejecución; `Desv. por flujo` es la media
de la desviación típica dentro de cada ejecución (jitter entre flujos).

| Dataset | Modelo | Media | Desv. entre ejec. | Desv. por flujo | Mediana | p95 | p99 | Máx |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| NF-UNSW-NB15 | SAD | 0,0051 | 0,0000 | 0,0011 | 0,0044 | 0,0063 | 0,0068 | 0,0745 |
| NF-UNSW-NB15 | LODA | 0,0936 | 0,0010 | 0,0288 | 0,0698 | 0,1270 | 0,1285 | 0,2794 |
| NF-UNSW-NB15 | One-Class SVM | 0,0021 | 0,0000 | 0,0017 | 0,0005 | 0,0039 | 0,0043 | 0,0320 |
| NF-UNSW-NB15-v2 | SAD | 0,0052 | 0,0002 | 0,0011 | 0,0059 | 0,0063 | 0,0068 | 0,0577 |
| NF-UNSW-NB15-v2 | LODA | 0,0556 | 0,0008 | 0,0160 | 0,0427 | 0,0734 | 0,0762 | 0,2957 |
| NF-UNSW-NB15-v2 | One-Class SVM | 0,0021 | 0,0000 | 0,0017 | 0,0026 | 0,0039 | 0,0041 | 0,0243 |
| NF-CSE-CIC-IDS2018 | SAD | 0,0052 | 0,0001 | 0,0023 | 0,0059 | 0,0063 | 0,0067 | 1,4575 |
| NF-CSE-CIC-IDS2018 | LODA | 0,0951 | 0,0037 | 0,0289 | 0,0921 | 0,1277 | 0,1293 | 1,5225 |
| NF-CSE-CIC-IDS2018 | One-Class SVM | 0,0024 | 0,0000 | 0,0016 | 0,0036 | 0,0038 | 0,0041 | 0,0619 |

La latencia de evaluación incluye, por flujo: construcción del registro,
`score_one`, comparación con el umbral (o `classify`) y la actualización online
del modelo cuando el flujo no se marca como anómalo. En One-Class SVM esa
actualización incluye el mantenimiento del cuantil del `QuantileFilter`. No
incluye codificación IP, escalado (se aplican en lote antes del bucle), parseo
del CSV, exportación/recolección NetFlow ni el cálculo de métricas y curvas.

## 5. Comandos para reproducir

**SAD**

```bash
# UNSW v1
.venv/bin/python sad_detector.py --filename ../NF-UNSW-NB15.csv \
  --seed <s> --aggregation max --threshold-step 0.1 --output-json salida.json

# UNSW v2
.venv/bin/python sad_detector.py --filename ../NF-UNSW-NB15-v2.csv \
  --seed <s> --aggregation max --threshold-step 0.1 --output-json salida.json

# CIC-IDS2018
.venv/bin/python sad_detector.py \
  --filename /tmp/opencode/NF-CSE-CIC-IDS2018-shuffled.csv \
  --seed <s> --aggregation max --threshold-step 0.5 --output-json salida.json
```

**LODA**

```bash
# UNSW v1
.venv/bin/python loda_detector.py --filename ../NF-UNSW-NB15.csv \
  --seed <s> --scaling-type Normalizer --n-bins 5 --n-random-cuts 100 \
  --output-json salida.json

# UNSW v2
.venv/bin/python loda_detector.py --filename ../NF-UNSW-NB15-v2.csv \
  --seed <s> --scaling-type StandardScaler --n-bins 20 --n-random-cuts 50 \
  --output-json salida.json

# CIC-IDS2018
.venv/bin/python loda_detector.py \
  --filename /tmp/opencode/NF-CSE-CIC-IDS2018-shuffled.csv \
  --seed <s> --scaling-type Normalizer --n-bins 5 --n-random-cuts 100 \
  --output-json salida.json
```

**One-Class SVM**

```bash
# UNSW v1 y v2 (12 semillas cada uno)
.venv/bin/python multiexecution_sad_loda.py --models OCSVM \
  --datasets NF-UNSW-NB15 NF-UNSW-NB15-v2

# CIC-IDS2018
.venv/bin/python ocsvm_cic.py \
  --filename /tmp/opencode/NF-CSE-CIC-IDS2018-shuffled.csv \
  --seed <s> --scaling-type StandardScaler --nu 0.5 --q 0.9 --learning-rate 0.3 \
  --output-json salida.json
```

**Nota importante:** los tres modelos usan semillas `0–11` en los tres datasets.
En UNSW la partición es posicional sobre el CSV y la semilla solo cambia el
orden interno (y las proyecciones de LODA); en CIC la partición viene fijada por
el CSV pre-barajado con `random.seed(42)`. Para regenerar todos los artefactos:

```bash
.venv/bin/python multiexecution_sad_loda.py --num-executions 12
.venv/bin/python multiexecution_cic.py --num-executions 12
```
