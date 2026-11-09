# Modelo LODA sobre NF-ToN-IoT (entrenamiento directo con benignos)

Prueba de `river.anomaly.LODA` sobre NF-ToN-IoT con el mismo protocolo que en
NF-UNSW-NB15: entrenamiento directo con 100.000 flujos benignos, umbral
calibrado con cuantiles de los scores de entrenamiento y evaluación con el
detector congelado.

ToN-IoT está muy desbalanceado (270.279 benignos frente a 1.108.995 anomalías),
así que el split balanceado del pipeline no cabe con todos los benignos. Se usa
un **subconjunto equilibrado**: todos los benignos y 169.279 anomalías, que son
las que permiten construir validación y test balanceados después de reservar
1.000 benignos para el escalador y 100.000 para entrenamiento.

## Metadatos

| Campo | Valor |
|---|---|
| Fecha de ejecución | 2026-10-28 |
| Rama | `feature/more-models` |
| Commit | `7d7e8a3` + cambios locales (`loda_detector.py`, documentos de resultados) |
| Script | `loda_detector.py` |
| Ejecuciones | 3: Standard 100 cortes, Standard 200 cortes y MaxAbs 100 cortes |
| Semilla | `42` (subconjunto, split y LODA) |
| Python | 3.12.3 |
| river | 0.26.1 |
| pandas | 3.1.0 |
| numpy | 2.5.4 |
| scikit-learn | 1.9.1 |
| matplotlib | 3.11.2 |
| PyYAML | 6.0.3 |

## Dataset y subconjunto

| Dataset | Flujos totales | Benignos | Anomalías | Tamaño |
|---|---:|---:|---:|---:|
| NF-ToN-IoT | 1.379.274 | 270.279 | 1.108.995 | 94 MB |

Distribución de ataques en el fichero original:

| Ataque | Flujos |
|---|---:|
| injection | 468.539 |
| ddos | 326.345 |
| password | 156.299 |
| xss | 99.944 |
| scanning | 21.467 |
| dos | 17.717 |
| backdoor | 17.247 |
| mitm | 1.295 |
| ransomware | 142 |
| **Benign** | **270.279** |

Subconjunto utilizado:

| Conjunto | Flujos |
|---|---:|
| Benignos | 270.279 (todos) |
| Anomalías | 169.279 (muestra aleatoria, `random_state=42`) |
| **Total** | **439.558** |

Particiones resultantes:

| Conjunto | Flujos |
|---|---:|
| Escalador | 1.000 benignos |
| Entrenamiento | 100.000 benignos |
| Validación | 84.640 benignos + 84.640 anomalías |
| Test | 84.639 benignos + 84.639 anomalías |

Características (`FEATURES` de `config.yaml`):

```text
IPV4_SRC_ADDR, IPV4_DST_ADDR, L4_DST_PORT, L4_SRC_PORT, PROTOCOL,
IN_BYTES, OUT_BYTES, FLOW_DURATION_MILLISECONDS, Label
```

## Configuración

| Parámetro | Valor |
|---|---|
| `FILENAME` | `/tmp/opencode/NF-ToN-IoT-balanced-shuffled.csv` |
| `TYPE_IP` | `IPv4` |
| `SIGNIFICANT_PORT` | `False` |
| `BENIGN_FLOWS_ADDITIONAL` | `0` |
| `VALIDATION_RATIO` | `0.5` |
| `FLOWS_TRAIN_SCALER` | `1000` |
| `FLOWS_TRAIN_OML` | `100000` |
| `scaling_type` | `StandardScaler` (y `MaxAbsScaler` en una variante) |
| `n_bins` | `10` |
| `n_random_cuts` | `100` (y `200` en una variante) |
| `seed` | `42` |
| `SELECTION_METRIC` | `f1` |
| Rejilla de cuantiles | `0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99` |
| Umbral | cuantil `q` de los scores de entrenamiento |
| Evaluación | detector congelado; test evaluado una sola vez |

## Resultados

### StandardScaler, 100 cortes

Cuantil seleccionado en validación: **q = 0,60** (umbral `1,225915`).

| Conjunto | Accuracy | Recall | FPR | Precision | F1 | TP / TN / FP / FN |
|---|---:|---:|---:|---:|---:|---:|
| Validación | 79,1753 % | 100,0000 % | 41,6493 % | 70,5969 % | 82,7646 % | 84.640 / 49.388 / 35.252 / 0 |
| **Test (held-out)** | **79,3015 %** | **99,9988 %** | **41,3958 %** | **70,7232 %** | **82,8509 %** | 84.638 / 49.602 / 35.037 / 1 |

Top 5 cuantiles en validación:

| q | Umbral | Accuracy | Recall | FPR | Precision | F1 |
|---:|---:|---:|---:|---:|---:|---:|
| **0,60** | **1,225915** | **79,1753 %** | **100,0000 %** | **41,6493 %** | **70,5969 %** | **82,7646 %** |
| 0,50 | 1,109117 | 74,8068 % | 100,0000 % | 50,3863 % | 66,4954 % | 79,8766 % |
| 0,70 | 1,471662 | 69,9604 % | 70,6486 % | 30,7278 % | 69,6894 % | 70,1657 % |
| 0,95 | 1,924041 | 51,7539 % | 8,6685 % | 5,1607 % | 62,6826 % | 15,2307 % |
| 0,80 | 1,656044 | 45,0337 % | 9,8169 % | 19,7495 % | 33,2028 % | 15,1534 % |

Tiempos: entrenamiento 6,04 s; scoring de validación 12,31 s; búsqueda del
umbral 0,07 s; scoring de test 12,23 s; **0,07 ms/flujo**.

### Variantes

| Configuración | q | Umbral | Accuracy | Recall | FPR | Precision | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Standard, 100 cortes | 0,60 | 1,225915 | 79,3015 % | 99,9988 % | 41,3958 % | 70,7232 % | 82,8509 % |
| Standard, 200 cortes | 0,60 | 1,285045 | 79,0776 % | 99,9988 % | 41,8436 % | 70,4999 % | 82,6975 % |
| **MaxAbs, 100 cortes** | **0,60** | **−1,236051** | **81,2191 %** | **99,9988 %** | **37,5607 %** | **72,6949 %** | **84,1884 %** |

`MaxAbsScaler` es la mejor configuración en ToN-IoT: mejora el F1 en ~1,34
puntos y reduce el FPR en ~3,84 puntos respecto a `StandardScaler`. Aun así, el
FPR sigue siendo muy alto.

Tiempos de las variantes:

| Configuración | Entrenamiento | Scoring de test | Coste por flujo |
|---|---:|---:|---:|
| Standard, 100 cortes | 6,04 s | 12,23 s | 0,07 ms |
| Standard, 200 cortes | 12,00 s | 23,68 s | 0,14 ms |
| MaxAbs, 100 cortes | 6,10 s | 11,99 s | 0,07 ms |

## Comparación con otros datasets

| Dataset | Modelo / split | Accuracy | Recall | FPR | Precision | F1 |
|---|---|---:|---:|---:|---:|---:|
| NF-ToN-IoT | LODA, MaxAbs, 100 cortes | 81,2191 % | 99,9988 % | 37,5607 % | 72,6949 % | 84,1884 % |
| NF-CSE-CIC-IDS2018 | LODA, split aleatorio | 81,9831 % | 94,1362 % | 30,1701 % | 75,7292 % | 83,9354 % |
| NF-UQ-NIDS | LODA, split aleatorio | 65,8883 % | 81,9697 % | 50,1930 % | 62,0218 % | 70,6140 % |
| NF-UNSW-NB15 | LODA, 100 cortes | 94,9175 % | 100,0000 % | 10,1649 % | 90,7730 % | 95,1634 % |
| NF-UNSW-NB15-v2 | LODA, 100 cortes | 98,6829 % | 100,0000 % | 2,6343 % | 97,4333 % | 98,7000 % |

Los valores de CIC, UQ y UNSW proceden de `resultados_loda_cic_ids2018.md`,
`resultados_loda_uq_nids.md`, `resultados_loda_unsw.md` y
`resultados_loda_unsw_v2.md`. No son ejecuciones idénticas: distinto dataset,
split y número de repeticiones.

## Observaciones

1. **ToN-IoT también es difícil para LODA.** El mejor resultado es ~81,22 % de
   accuracy, ~84,19 % de F1 y ~37,56 % de FPR, con recall ~100 %. No cumple los
   objetivos del paper (accuracy > 98 %, FPR < 3,1 %, recall 100 %).
2. **El subconjunto es necesario por el desbalanceo.** Con 270.279 benignos no
   se pueden construir validación y test balanceados para 1.108.995 anomalías.
   Se usan todos los benignos y 169.279 anomalías; la evaluación es 50/50, no
   representa la prevalencia real de ToN-IoT (~80 % de anomalías).
3. **MaxAbsScaler ayuda en ToN-IoT.** A diferencia de UNSW v1 y CIC, aquí
   `MaxAbsScaler` supera a `StandardScaler` (F1 84,19 % vs 82,85 %), aunque el
   FPR sigue siendo ~37,6 %.
4. **El cuantil óptimo es 0,60 en todas las variantes.** Con q = 0,50 el recall
   sigue en 100 % pero el FPR sube al ~45,6 % (MaxAbs); con q = 0,70 el recall
   cae al ~57,9 % (MaxAbs). No hay un umbral que dé recall alto y FPR bajo.
5. **Más cortes no ayudan.** Pasar de 100 a 200 cortes empeora ligeramente el
   F1 (82,85 % → 82,70 %) y duplica el coste.
6. **Una sola ejecución por configuración.** Los resultados corresponden a un
   subconjunto fijo (semilla 42); no se reporta media ± desviación típica.
7. **Precisión y FPR dependen del balance 50/50.** En tráfico real, con menos
   anomalías, la precisión sería menor.

## Reproducibilidad

Crear el subconjunto equilibrado y barajado:

```python
import pandas as pd

df = pd.read_csv('../NF-ToN-IoT.csv')
benign = df[df['Label'] == 0]
anomalous = df[df['Label'] == 1]

# 1.000 benignos para el escalador + 100.000 para entrenamiento.
n_anomalies = len(benign) - 1_000 - 100_000  # 169.279
subset = pd.concat([
    benign,
    anomalous.sample(n=n_anomalies, random_state=42),
])
subset = subset.sample(frac=1, random_state=42)
subset.to_csv('/tmp/opencode/NF-ToN-IoT-balanced-shuffled.csv', index=False)
```

Ejecutar:

```bash
cd "/home/alberto/Online machine learning/codigo-paper-oml-computer-security"
.venv/bin/python loda_detector.py \
  --filename /tmp/opencode/NF-ToN-IoT-balanced-shuffled.csv

.venv/bin/python loda_detector.py \
  --filename /tmp/opencode/NF-ToN-IoT-balanced-shuffled.csv \
  --n-random-cuts 200

.venv/bin/python loda_detector.py \
  --filename /tmp/opencode/NF-ToN-IoT-balanced-shuffled.csv \
  --scaling-type MaxAbsScaler
```

Dependencias: `river`, `pandas`, `numpy`, `scikit-learn`, `matplotlib` y `pyyaml`
(las tres últimas porque `loda_detector.py` reutiliza `hyperparameter_optimization.py`,
que las importa a nivel de módulo).

El script fija `np.random.seed(42)` antes de generar las particiones y usa
`seed=42` en LODA. Por defecto no escribe ningún fichero de métricas; si se
quiere una copia JSON, usar `--output-json`.
