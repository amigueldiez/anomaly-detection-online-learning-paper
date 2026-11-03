# Modelo LODA sobre NF-UNSW-NB15 (entrenamiento directo con benignos)

Prueba funcional del detector `river.anomaly.LODA` sobre el dataset NF-UNSW-NB15
(v1), entrenándolo **directamente con flujos benignos** y calibrando después el
umbral con los scores de entrenamiento. Se usa el mismo esquema de particiones
que `hyperparameter_optimization.py`, pero se evita `QuantileFilter` durante el
warm-up para no congelar el detector.

## Metadatos

| Campo | Valor |
|---|---|
| Fecha de ejecución | 2026-10-27 |
| Rama | `feature/more-models` |
| Commit | `7d7e8a3` + cambios locales (`loda_unsw.py`, este documento) |
| Script | `loda_unsw.py` |
| Ejecuciones | 1 configuración principal + 3 variantes |
| Semilla | `42` (split y LODA) |
| Python | 3.12.3 |
| river | 0.26.1 |
| pandas | 3.1.0 |
| numpy | 2.5.4 |
| scikit-learn | 1.9.1 |
| matplotlib | 3.11.2 |
| PyYAML | 6.0.3 |

## Motivación: por qué se entrena LODA directamente

`hyperparameter_optimization.py` envuelve el detector en
`anomaly.QuantileFilter(..., protect_anomaly_detector=True)` desde la primera
muestra. Ese filtro calcula el score **antes** de actualizar el detector:

1. LODA devuelve su score máximo (27,63) cuando un valor cae fuera de todos sus
   histogramas.
2. El cuantil del filtro salta a ese score máximo.
3. Todas las muestras siguientes se clasifican como anómalas.
4. Como `protect_anomaly_detector=True`, el detector deja de aprender.

En una comprobación con las primeras 2.000 muestras benignas, el filtro solo
dejó aprender **2 muestras** con `StandardScaler` (frente a 1.968 con
`MaxAbsScaler`). El detector quedaba congelado y marcaba todo como anómalo.
Por eso este script:

- entrena LODA directamente con los 100.000 flujos benignos;
- calcula los scores de esos mismos flujos benignos;
- usa cuantiles de esos scores como umbrales candidatos;
- elige el umbral en validación y evalúa test una sola vez.

Durante la evaluación el detector **no se actualiza** (`score_one` solamente).
Así todos los umbrales se comparan sobre el mismo modelo y no hay realimentación
entre el umbral y el aprendizaje.

## Modelo

LODA (*Lightweight On-line Detector of Anomalies*) es un ensamble de histogramas
unidimensionales construidos sobre proyecciones aleatorias dispersas de las
características. Para cada proyección se estima la densidad del valor proyectado;
el score final es la media de las log-verosimilitudes negativas. Valores raros
producen densidades bajas y, por tanto, scores altos.

| Parámetro | Valor |
|---|---|
| `n_bins` | `10` |
| `n_random_cuts` | `100` (por defecto del script) |
| `seed` | `42` |
| Características | 8 (las mismas de `config.yaml` sin `Label`) |
| Entrenamiento | solo flujos benignos |
| Evaluación | detector congelado; umbral fijo |

## Datos y particiones

Características (`FEATURES` de `config.yaml`):

```text
IPV4_SRC_ADDR, IPV4_DST_ADDR, L4_DST_PORT, L4_SRC_PORT, PROTOCOL,
IN_BYTES, OUT_BYTES, FLOW_DURATION_MILLISECONDS, Label
```

| Dataset | Flujos totales | Benignos | Anomalías | Escalador | Entrenamiento | Validación (benignos/anomalías) | Test (benignos/anomalías) |
|---|---:|---:|---:|---:|---:|---:|---:|
| NF-UNSW-NB15 | 1.623.118 | 1.550.712 | 72.406 | 1.000 | 100.000 | 36.203 / 36.203 | 36.203 / 36.203 |

Parámetros comunes:

| Parámetro | Valor |
|---|---|
| `FILENAME` | `../NF-UNSW-NB15.csv` |
| `TYPE_IP` | `IPv4` |
| `SIGNIFICANT_PORT` | `False` |
| `BENIGN_FLOWS_ADDITIONAL` | `0` |
| `VALIDATION_RATIO` | `0.5` |
| `FLOWS_TRAIN_SCALER` | `1000` |
| `FLOWS_TRAIN_OML` | `100000` |
| `scaling_type` | `StandardScaler` |
| `SELECTION_METRIC` | `f1` |
| Rejilla de cuantiles | `0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99` |
| Umbral | cuantil `q` de los scores de entrenamiento |
| Actualización durante validación/test | ninguna |

## Cómo ejecutar

Desde la raíz del repositorio:

```bash
# Configuración principal (StandardScaler, 100 cortes)
.venv/bin/python loda_unsw.py

# Variantes documentadas
.venv/bin/python loda_unsw.py --n-random-cuts 200
.venv/bin/python loda_unsw.py --n-random-cuts 500
.venv/bin/python loda_unsw.py --scaling-type MaxAbsScaler --n-random-cuts 200
```

Opciones disponibles:

| Opción | Por defecto | Descripción |
|---|---|---|
| `--filename` | `../NF-UNSW-NB15.csv` | CSV etiquetado a procesar |
| `--flows-train-oml` | `100000` | Flujos benignos de entrenamiento |
| `--scaling-type` | `StandardScaler` | `MaxAbsScaler`, `MinMaxScaler`, `Normalizer`, `RobustScaler` o `StandardScaler` |
| `--n-bins` | `10` | Número máximo de bins por histograma |
| `--n-random-cuts` | `100` | Número de proyecciones aleatorias |
| `--q-grid` | `0.5,0.6,0.7,0.8,0.85,0.9,0.95,0.99` | Cuantiles candidatos |
| `--selection-metric` | `f1` | `accuracy`, `precision`, `recall`, `false_positive` o `f1` |
| `--seed` | `42` | Semilla del split y de LODA |
| `--output-json` | desactivado | Ruta opcional para guardar las métricas en JSON |

## Resultados

### Configuración principal: `StandardScaler`, 100 cortes

Cuantil seleccionado en validación: **q = 0,90** (umbral `1,077823`).

| Conjunto | Accuracy | Recall | FPR | Precision | F1 | TP / TN / FP / FN |
|---|---:|---:|---:|---:|---:|---:|
| Validación | 95,8733 % | 100,0000 % | 8,2535 % | 92,3758 % | 96,0368 % | 36.203 / 33.215 / 2.988 / 0 |
| **Test (held-out)** | **94,9175 %** | **100,0000 %** | **10,1649 %** | **90,7730 %** | **95,1634 %** | 36.203 / 32.523 / 3.680 / 0 |

Los cinco mejores cuantiles en validación fueron:

| q | Umbral | Accuracy | Recall | FPR | Precision | F1 |
|---:|---:|---:|---:|---:|---:|---:|
| **0,90** | **1,077823** | **95,8733 %** | **100,0000 %** | **8,2535 %** | **92,3758 %** | **96,0368 %** |
| 0,85 | 0,994079 | 93,0351 % | 100,0000 % | 13,9298 % | 87,7734 % | 93,4886 % |
| 0,80 | 0,957757 | 90,2619 % | 100,0000 % | 19,4763 % | 83,6986 % | 91,1260 % |
| 0,70 | 0,902884 | 85,5081 % | 100,0000 % | 28,9838 % | 77,5291 % | 87,3424 % |
| 0,60 | 0,833423 | 79,6909 % | 100,0000 % | 40,6182 % | 71,1146 % | 83,1192 % |

Tiempos:

| Fase | Tiempo |
|---|---:|
| Entrenamiento (100.000 flujos) | 5,99 s |
| Scoring del entrenamiento (100.000) | 7,08 s |
| Scoring de validación (72.406) | 5,13 s |
| Búsqueda del umbral (8 cuantiles) | 0,03 s |
| Scoring de test (72.406) | 5,10 s |
| Coste por flujo en test | 0,07 ms/flujo |

### Variantes probadas

Resultados de validación:

| Configuración | q | Umbral | Accuracy | Recall | FPR | Precision | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Standard, 100 cortes | 0,90 | 1,077823 | 95,8733 % | 100,0000 % | 8,2535 % | 92,3758 % | 96,0368 % |
| Standard, 200 cortes | 0,90 | 1,047319 | 95,5266 % | 100,0000 % | 8,9468 % | 91,7879 % | 95,7182 % |
| Standard, 500 cortes | 0,90 | 1,132989 | 95,6150 % | 100,0000 % | 8,7700 % | 91,9371 % | 95,7992 % |
| MaxAbs, 200 cortes | 0,85 | −2,375593 | 92,2189 % | 94,9396 % | 10,5019 % | 90,0401 % | 92,4250 % |

Resultados de test:

| Configuración | Accuracy | Recall | FPR | Precision | F1 |
|---|---:|---:|---:|---:|---:|
| **Standard, 100 cortes** | **94,9175 %** | **100,0000 %** | **10,1649 %** | **90,7730 %** | **95,1634 %** |
| Standard, 200 cortes | 94,4397 % | 100,0000 % | 11,1206 % | 89,9923 % | 94,7326 % |
| Standard, 500 cortes | 94,4880 % | 100,0000 % | 11,0239 % | 90,0707 % | 94,7760 % |
| MaxAbs, 200 cortes | 91,7479 % | 95,0170 % | 11,5211 % | 89,1859 % | 92,0091 % |

Tiempos de las variantes:

| Configuración | Entrenamiento | Scoring de test | Coste por flujo |
|---|---:|---:|---:|
| Standard, 100 cortes | 5,99 s | 5,10 s | 0,07 ms |
| Standard, 200 cortes | 11,80 s | 10,19 s | 0,14 ms |
| Standard, 500 cortes | 30,20 s | 26,12 s | 0,36 ms |
| MaxAbs, 200 cortes | 11,98 s | 10,24 s | 0,14 ms |

## Comparación con otros modelos

Valores de test de referencia (no son ejecuciones idénticas: distinto protocolo,
split y número de repeticiones):

| Modelo | Protocolo | Accuracy | Recall | FPR | F1 |
|---|---|---:|---:|---:|---:|
| One-Class SVM | `hyperparameter_optimization.py`, media de 12 ejecuciones | 97,7703 % | 99,5007 % | 3,9601 % | 97,8081 % |
| SAD | `sad_unsw.py`, 1 ejecución, umbral 4,9σ | 97,2143 % | 100,0000 % | 5,5714 % | 97,2898 % |
| **LODA** | `loda_unsw.py`, 1 ejecución, 100 cortes | 94,9175 % | **100,0000 %** | 10,1649 % | 95,1634 % |

Los valores de One-Class SVM proceden de
`resultados_split_entrenamiento_validacion_test.md`; los de SAD, de
`resultados_sad_unsw.md`.

## Observaciones

1. **LODA detecta todas las anomalías en test, pero con un FPR alto.** Con
   `StandardScaler` y 100 cortes se obtiene recall 100 % y F1 95,1634 %, pero el
   FPR es 10,1649 %, por encima del objetivo del paper (< 3,1 %).
2. **100 cortes es la mejor variante en esta partición y la más rápida.**
   Aumentar a 200 o 500 cortes no mejora el F1 de test (94,73 % y 94,78 %) y
   multiplica el coste. En pruebas exploratorias con otra permutación del split,
   200/500 cortes dieron F1 ≈ 95,6 %, por lo que la diferencia es pequeña y
   depende del orden de entrenamiento.
3. **`StandardScaler` es claramente mejor que `MaxAbsScaler`.** Con MaxAbs el
   recall baja al 95,0170 % y el F1 al 92,0091 %, además de necesitar un cuantil
   más bajo (0,85) para compensar la separación entre clases.
4. **El cuantil óptimo es 0,90 para las tres configuraciones con
   `StandardScaler`.** Por debajo de 0,90 el recall se mantiene en 100 %, pero el
   FPR sube rápidamente (13,9 % con q = 0,85 y 19,5 % con q = 0,80).
5. **LODA queda por debajo de One-Class SVM y SAD en este dataset.** One-Class
   SVM consigue ~97,8 % de F1 con ~4,0 % de FPR; SAD, ~97,3 % de F1 con ~5,6 %
   de FPR. LODA iguala el recall, pero a costa de duplicar o triplicar el FPR.
6. **Una sola ejecución por configuración.** Los resultados corresponden a una
   partición fija (semilla 42). Para reportar media ± desviación típica habría
   que repetir con varias semillas, como hace `multiexecution.py`.
7. **Validación y test están balanceados al 50 %.** El FPR y el recall deben
   interpretarse con esa proporción; en tráfico real, mucho más desbalanceado,
   la precisión sería menor.

## Reproducibilidad

```bash
cd "/home/alberto/Online machine learning/codigo-paper-oml-computer-security"

# Configuración principal
.venv/bin/python loda_unsw.py

# Variantes
.venv/bin/python loda_unsw.py --n-random-cuts 200
.venv/bin/python loda_unsw.py --n-random-cuts 500
.venv/bin/python loda_unsw.py --scaling-type MaxAbsScaler --n-random-cuts 200
```

Dependencias que deben estar en el entorno virtual: `river`, `pandas`, `numpy`,
`scikit-learn`, `matplotlib` y `pyyaml` (estas tres últimas porque
`loda_unsw.py` reutiliza `hyperparameter_optimization.py`, que las importa a nivel
de módulo).

El script fija `np.random.seed(42)` antes de generar las particiones y usa
`seed=42` en LODA, por lo que las tablas se reproducen con las mismas versiones
de las librerías. Por defecto no escribe ningún fichero de métricas; si se
quiere una copia JSON, usar `--output-json`.

## Otros datasets

La misma prueba sobre NF-UNSW-NB15-v2 está documentada en
`resultados_loda_unsw_v2.md`. Resumen: LODA mejora claramente en v2 (F1 de test
98,7000 %, recall 100 %, FPR 2,6343 %), cumpliendo los objetivos del paper,
aunque sigue por debajo de One-Class SVM y SAD.

La prueba sobre NF-CSE-CIC-IDS2018 está documentada en
`resultados_loda_cic_ids2018.md`. Resumen: resultado negativo; incluso con
split aleatorio, LODA se queda en ~83,94 % de F1 y ~30,17 % de FPR.

Las pruebas sobre NF-ToN-IoT y NF-UQ-NIDS están documentadas en
`resultados_loda_ton_iot.md` y `resultados_loda_uq_nids.md`. Resumen: ToN-IoT
llega a ~84,19 % de F1 con ~37,56 % de FPR; UQ-NIDS se queda en ~70,61 % de F1
con ~50,19 % de FPR.

## Repeticiones multisemilla en UNSW

Los resultados de este documento corresponden a una sola ejecución. La
evaluación con 12 semillas sobre UNSW v1 y v2 está en
`resultados_multisemilla_sad_loda_unsw.md`. Resumen de test (media ± desviación
típica, n = 12):

| Dataset | Accuracy | Recall | FPR | F1 |
|---|---:|---:|---:|---:|
| NF-UNSW-NB15 | 95,0337 ± 0,4223 % | 99,6962 ± 1,0525 % | 9,6288 ± 1,3981 % | 95,2561 ± 0,3819 % |
| NF-UNSW-NB15-v2 | 98,8770 ± 0,3137 % | 99,9998 ± 0,0006 % | 2,2459 ± 0,6279 % | 98,8903 ± 0,3091 % |
