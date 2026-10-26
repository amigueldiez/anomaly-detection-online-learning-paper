# Modelo StandardAbsoluteDeviation (SAD) sobre NF-UNSW-NB15

Prueba funcional del detector `river.anomaly.StandardAbsoluteDeviation` aplicado a
los datasets NF-UNSW-NB15 y NF-UNSW-NB15-v2 con el mismo esquema de particiones
del pipeline principal (`hyperparameter_optimization.py`).

## Metadatos

| Campo | Valor |
|---|---|
| Fecha de ejecución | 2026-10-26 |
| Rama | `feature/more-models` |
| Commit | `ce381b0` |
| Script | `sad_unsw.py` |
| Ejecuciones | 1 por dataset, semilla fija `42` |
| Python | 3.12.3 |
| river | 0.26.1 |
| pandas | 3.1.0 |
| numpy | 2.5.4 |
| scikit-learn | 1.9.1 |
| matplotlib | 3.11.2 |
| PyYAML | 6.0.3 |

## Cómo funciona el modelo

### SAD es un detector univariante y no supervisado

`StandardAbsoluteDeviation` calcula, para cada valor observado `y`, cuántas
desviaciones típicas se aleja de la media (o mediana) de la serie:

```text
score = |y - sub_stat| / sqrt(varianza)
```

- `sub_stat` es `stats.Mean()` por defecto (también admite `stats.Median()`).
- La varianza es la varianza poblacional acumulada (`stats.Var`).
- Un `score` de `2` significa "2σ"; el umbral clásico de la regla 3σ es `3`.

El modelo **no usa etiquetas**: solo aprende de los valores que observa. En River
hereda de `SupervisedAnomalyDetector` porque su API es `learn_one(x, y)` y
`score_one(x, y)`, donde `y` es el valor escalar a analizar. La `x` se ignora.
Ese "supervised" hace referencia a la firma univariante, no a que use la clase
`Label`.

### Adaptación a flujos multivariantes: un SAD por característica

Un flujo de red tiene 8 características, así que se mantiene **una instancia SAD
por característica**. Para cada flujo se obtienen 8 scores y se combinan en uno
solo con una agregación:

```text
flujo → [z(IN_BYTES), z(OUT_BYTES), z(L4_DST_PORT), ...] → max(...) → score
```

La agregación usada es `max`: si **alguna** característica se sale de lo normal,
el flujo se considera sospechoso. Se descartan las características con varianza
cero tras el warm-up, porque su score explotaría (división por ~0).

Se probaron también `mean` y `median`, pero diluyen el ataque entre las
características que no cambian: con `max` se obtuvo recall 100 % en v2, mientras
que con `mean`/`median` el recall caía prácticamente a 0 %.

### Aprendizaje online

Durante el warm-up el modelo aprende solo con flujos benignos. Después, en cada
flujo nuevo:

1. Se calcula el score y se compara con el umbral.
2. Si **no** se marca como anómalo, el modelo se actualiza con ese flujo.
3. Si se marca como anómalo, no se aprende de él.

Así el modelo sigue la distribución benigna sin contaminarse con los ataques que
detecta.

### Selección del umbral (sin fuga de información)

El umbral no se elige mirando el test. El protocolo es:

1. Ajustar el escalador y entrenar SAD con 100.000 flujos benignos.
2. Para cada umbral candidato (0,0–10,0σ en pasos de 0,1, más 12, 15 y 20 =
   104 candidatos), reentrenar desde cero y evaluar **online** sobre validación.
3. Elegir el umbral que maximiza `SELECTION_METRIC` (por defecto `f1`). Si hay
   varios empatados, se toma la **mediana del plateau**, para no dejar el umbral
   pegado al score mínimo de las anomalías ni al máximo de los benignos.
4. Reentrenar desde cero y evaluar el test **una sola vez** con ese umbral.

Validación se usa solo para elegir el umbral; test permanece Held-out.

## Datos y particiones

Características (`FEATURES` de `config.yaml`):

```text
IPV4_SRC_ADDR, IPV4_DST_ADDR, L4_DST_PORT, L4_SRC_PORT, PROTOCOL,
IN_BYTES, OUT_BYTES, FLOW_DURATION_MILLISECONDS, Label
```

Particiones (idénticas al pipeline principal):

| Dataset | Flujos totales | Benignos | Anomalías | Escalador | Entrenamiento | Validación (benignos/anomalías) | Test (benignos/anomalías) |
|---|---:|---:|---:|---:|---:|---:|---:|
| NF-UNSW-NB15 | 1.623.118 | 1.550.712 | 72.406 | 1.000 | 100.000 | 36.203 / 36.203 | 36.203 / 36.203 |
| NF-UNSW-NB15-v2 | 2.390.275 | 2.295.222 | 95.053 | 1.000 | 100.000 | 47.526 / 47.526 | 47.527 / 47.527 |

Parámetros comunes:

| Parámetro | Valor |
|---|---|
| Escalador | `MaxAbsScaler` |
| Agregación | `max` |
| Warm-up (`FLOWS_TRAIN_OML`) | 100.000 flujos benignos |
| `FLOWS_TRAIN_SCALER` | 1.000 |
| Métrica de selección | `f1` |
| Semilla del split | `42` |

## Cómo ejecutar los experimentos

Desde la raíz del repositorio (el script lee `config.yaml` y rutas relativas):

```bash
# NF-UNSW-NB15 (config.yaml)
.venv/bin/python sad_unsw.py

# NF-UNSW-NB15-v2
.venv/bin/python sad_unsw.py --filename ../NF-UNSW-NB15-v2.csv
```

Opciones disponibles:

| Opción | Por defecto | Descripción |
|---|---|---|
| `--filename` | `../NF-UNSW-NB15.csv` | CSV etiquetado a procesar |
| `--flows-train-oml` | `100000` | Flujos benignos de warm-up |
| `--aggregation` | `max` | `max`, `mean` o `median` |
| `--selection-metric` | `f1` | `accuracy`, `precision`, `recall`, `false_positive` o `f1` |
| `--seed` | `42` | Semilla del split scaler/entrenamiento/validación/test |

> No se recomienda `--selection-metric false_positive` para elegir el umbral:
> como un umbral muy alto no marca ningún flujo, minimizar el FPR llevaría a no
> detectar nada. Usar `f1`, `recall` o `accuracy`.

Ejemplos:

```bash
# Cambiar la métrica de selección del umbral
.venv/bin/python sad_unsw.py --filename ../NF-UNSW-NB15-v2.csv --selection-metric recall

# Comparar agregaciones
.venv/bin/python sad_unsw.py --aggregation mean
```

Dependencias que deben estar en el entorno virtual: `river`, `pandas`, `numpy`,
`scikit-learn`, `matplotlib` y `pyyaml` (estas tres últimas porque
`sad_unsw.py` reutiliza `hyperparameter_optimization.py`, que las importa a nivel
de módulo).

Salida esperada: la partición, la búsqueda del umbral (con los 5 mejores
umbrales), las métricas de validación y las métricas finales de test, además de
los tiempos.

## Resultados

### NF-UNSW-NB15

Umbral seleccionado en validación: **4,9σ**.

| Conjunto | Accuracy | Recall | FPR | Precision | F1 | TP / TN / FP / FN |
|---|---:|---:|---:|---:|---:|---:|
| Validación | 97,6190 % | 100,0000 % | 4,7620 % | 95,4544 % | 97,6744 % | 36.203 / 34.479 / 1.724 / 0 |
| **Test (held-out)** | **97,2143 %** | **100,0000 %** | **5,5714 %** | **94,7227 %** | **97,2898 %** | 36.203 / 34.186 / 2.017 / 0 |

Tiempos:

| Fase | Tiempo |
|---|---:|
| Búsqueda del umbral (104 candidatos) | 61,45 s |
| Entrenamiento final (100.000 flujos) | 0,19 s |
| Evaluación de test | 0,38 s |

### NF-UNSW-NB15-v2

Umbral seleccionado en validación: **5,7σ** (plateau de 3,8–7,5σ con idéntico
F1; se eligió la mediana).

| Conjunto | Accuracy | Recall | FPR | Precision | F1 | TP / TN / FP / FN |
|---|---:|---:|---:|---:|---:|---:|
| Validación | 99,8159 % | 100,0000 % | 0,3682 % | 99,6331 % | 99,8162 % | 47.526 / 47.351 / 175 / 0 |
| **Test (held-out)** | **99,8874 %** | **100,0000 %** | **0,2251 %** | **99,7754 %** | **99,8876 %** | 47.527 / 47.420 / 107 / 0 |

Tiempos:

| Fase | Tiempo |
|---|---:|
| Búsqueda del umbral (104 candidatos) | 72,20 s |
| Entrenamiento final (100.000 flujos) | 0,19 s |
| Evaluación de test | 0,49 s |

### Comparación con el pipeline One-Class SVM

Valores de test del pipeline principal (`hyperparameter_optimization.py`, media
de 12 ejecuciones; ver `resultados_split_entrenamiento_validacion_test.md`):

| Dataset | Modelo | Accuracy | Recall | FPR | F1 |
|---|---|---:|---:|---:|---:|
| NF-UNSW-NB15 | One-Class SVM | 97,7703 % | 99,5007 % | 3,9601 % | 97,8081 % |
| NF-UNSW-NB15 | **SAD** | 97,2143 % | **100,0000 %** | 5,5714 % | 97,2898 % |
| NF-UNSW-NB15-v2 | One-Class SVM | 99,7514 % | 99,5210 % | **0,0182 %** | 99,7491 % |
| NF-UNSW-NB15-v2 | **SAD** | **99,8874 %** | **100,0000 %** | 0,2251 % | **99,8876 %** |

La comparación es orientativa: el split, el número de ejecuciones y el criterio
de umbral no son idénticos en ambos experimentos.

## Observaciones

1. **SAD funciona muy bien en v2.** Con el umbral elegido solo en validación se
   obtiene 99,8874 % de accuracy, 100 % de recall y 0,2251 % de FPR en test. Se
   cumple el objetivo del paper para v2 (accuracy > 98 %, FPR < 3,1 %).
2. **En v1 el FPR supera el 3,1 %.** El test queda en 5,5714 % de falsos
   positivos. La separación entre benignos y anomalías en v1 es peor que en v2;
   se ve también en que el plateau de umbrales óptimos es mucho más estrecho.
3. **La agregación importa.** Con `mean` o `median` el score se diluye y el
   recall cae a ~0 %. `max` es la opción adecuada cuando un ataque altera solo
   un subconjunto de características.
4. **El umbral es un parámetro crítico y depende de la validación.** En v2 el F1
   es idéntico para 3,8–7,5σ (38 candidatos), por lo que el umbral se elige en
   la mediana del plateau (5,7σ). En v1 el plateau es estrecho y el resultado es
   más sensible a la muestra de validación.
5. **Una sola ejecución por dataset.** Los resultados corresponden a una
   partición fija (semilla 42). Para reportar media ± desviación típica habría
   que repetir con semillas distintas, como hace `multiexecution.py` con el
   pipeline principal.
6. **Validación y test están balanceados al 50 %.** El FPR y el recall deben
   interpretarse con esa proporción; en tráfico real, mucho más desbalanceado,
   el número absoluto de falsos positivos por flujo benigno sería el mismo, pero
   la precisión cambiaría.

## Reproducibilidad

```bash
cd "/home/alberto/Online machine learning/codigo-paper-oml-computer-security"

# v1
.venv/bin/python sad_unsw.py

# v2
.venv/bin/python sad_unsw.py --filename ../NF-UNSW-NB15-v2.csv
```

Los ficheros de datos son:

- `NF-UNSW-NB15`: `/home/alberto/Online machine learning/NF-UNSW-NB15.csv`
- `NF-UNSW-NB15-v2`: `/home/alberto/Online machine learning/NF-UNSW-NB15-v2.csv`

El script fija `np.random.seed(42)` antes de generar las particiones, por lo que
las tablas de este documento se reproducen ejecutando los comandos anteriores
con las mismas versiones de las librerías.
