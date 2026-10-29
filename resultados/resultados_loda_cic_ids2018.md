# Modelo LODA sobre NF-CSE-CIC-IDS2018 (entrenamiento directo con benignos)

Prueba de `river.anomaly.LODA` sobre NF-CSE-CIC-IDS2018 v1 con el mismo
protocolo que en NF-UNSW-NB15: entrenamiento directo con 100.000 flujos
benignos, umbral calibrado con cuantiles de los scores de entrenamiento y
evaluación con el detector congelado.

El resultado es **negativo**: LODA no generaliza bien a este dataset. Se
documentan tanto el split secuencial del pipeline como un split aleatorio, para
separar el efecto del modelo del efecto del particionado.

## Metadatos

| Campo | Valor |
|---|---|
| Fecha de ejecución | 2026-10-27 |
| Rama | `feature/more-models` |
| Commit | `7d7e8a3` + cambios locales (`loda_unsw.py`, documentos de resultados) |
| Script | `loda_unsw.py` |
| Ejecuciones | 4: split secuencial + split aleatorio (100 cortes) + 200 cortes + MaxAbsScaler |
| Semilla | `42` (split y LODA) |
| Python | 3.12.3 |
| river | 0.26.1 |
| pandas | 3.1.0 |
| numpy | 2.5.4 |
| scikit-learn | 1.9.1 |
| matplotlib | 3.11.2 |
| PyYAML | 6.0.3 |

## Dataset y particiones

| Dataset | Flujos totales | Benignos | Anomalías | Tamaño |
|---|---:|---:|---:|---:|
| NF-CSE-CIC-IDS2018 (v1) | 8.392.399 | 7.373.196 | 1.019.203 | 592 MB |

| Conjunto | Flujos |
|---|---:|
| Escalador | 1.000 benignos |
| Entrenamiento | 100.000 benignos |
| Validación | 509.602 benignos + 509.602 anomalías |
| Test | 509.601 benignos + 509.601 anomalías |

Características (`FEATURES` de `config.yaml`):

```text
IPV4_SRC_ADDR, IPV4_DST_ADDR, L4_DST_PORT, L4_SRC_PORT, PROTOCOL,
IN_BYTES, OUT_BYTES, FLOW_DURATION_MILLISECONDS, Label
```

La columna `Attack` se descarta antes de construir las particiones.

## Configuración

| Parámetro | Valor |
|---|---|
| `FILENAME` | `../NF-CSE-CIC-IDS2018.csv` (secuencial) o CSV barajado (aleatorio) |
| `TYPE_IP` | `IPv4` |
| `SIGNIFICANT_PORT` | `False` |
| `BENIGN_FLOWS_ADDITIONAL` | `0` |
| `VALIDATION_RATIO` | `0.5` |
| `FLOWS_TRAIN_SCALER` | `1000` |
| `FLOWS_TRAIN_OML` | `100000` |
| `scaling_type` | `StandardScaler` |
| `n_bins` | `10` |
| `n_random_cuts` | `100` (y `200` en una variante) |
| `seed` | `42` |
| `SELECTION_METRIC` | `f1` |
| Rejilla de cuantiles | `0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99` |
| Umbral | cuantil `q` de los scores de entrenamiento |
| Evaluación | detector congelado; test evaluado una sola vez |

## Dos formas de partir el dataset

El pipeline (`generate_dataset`) construye validación y test con **bloques
consecutivos** del CSV. En CIC-IDS2018 los flujos están agrupados por capturas
y tipos de ataque, así que los bloques pueden tener distribuciones distintas.
Por eso se repiten los experimentos con dos particiones:

1. **Split secuencial**: comportamiento por defecto del pipeline.
2. **Split aleatorio**: se baraja el CSV completo con `random.seed(42)` y se
   aplica el mismo `generate_dataset` sobre el fichero barajado.

Para reproducir el split aleatorio:

```python
import random
from pathlib import Path

src = Path('../NF-CSE-CIC-IDS2018.csv')
dst = Path('/tmp/opencode/NF-CSE-CIC-IDS2018-shuffled.csv')

with src.open('r', newline='') as f:
    header = f.readline()
    rows = f.readlines()

random.seed(42)
random.shuffle(rows)

with dst.open('w', newline='') as f:
    f.write(header)
    f.writelines(rows)
```

El fichero barajado ocupa ~591 MB y no se guarda en el repositorio.

## Resultados

### Split secuencial (por defecto)

Cuantil seleccionado en validación: **q = 0,70** (umbral `1,302942`).

| Conjunto | Accuracy | Recall | FPR | Precision | F1 | TP / TN / FP / FN |
|---|---:|---:|---:|---:|---:|---:|
| Validación | 70,9746 % | 92,2534 % | 50,3042 % | 64,7131 % | 76,0672 % | 470.125 / 253.251 / 256.351 / 39.477 |
| **Test (held-out)** | **71,8578 %** | **100,0000 %** | **56,2844 %** | **63,9859 %** | **78,0383 %** | 509.601 / 222.775 / 286.826 / 0 |

Top 5 cuantiles en validación:

| q | Umbral | Accuracy | Recall | FPR | Precision | F1 |
|---:|---:|---:|---:|---:|---:|---:|
| **0,70** | **1,302942** | **70,9746 %** | **92,2534 %** | **50,3042 %** | **64,7131 %** | **76,0672 %** |
| 0,60 | 1,032170 | 65,1156 % | 93,7993 % | 63,5680 % | 59,6053 % | 72,8914 % |
| 0,50 | 0,904307 | 52,5894 % | 99,1491 % | 93,9704 % | 51,3408 % | 67,6510 % |
| 0,80 | 1,574377 | 58,9613 % | 47,3101 % | 29,3874 % | 61,6840 % | 53,5492 % |
| 0,85 | 1,629650 | 58,4603 % | 36,5014 % | 19,5808 % | 65,0856 % | 46,7721 % |

Tiempos: entrenamiento 6,62 s; scoring de validación 72,86 s; búsqueda del
umbral 0,44 s; scoring de test 72,46 s; **0,07 ms/flujo**.

La diferencia entre validación y test (el recall sube de 92,25 % a 100 % y el
FPR de 50,30 % a 56,28 %) indica que los bloques no representan la misma
distribución.

### Split aleatorio (`random.seed(42)`)

Cuantil seleccionado en validación: **q = 0,70** (umbral `1,222079`).

| Conjunto | Accuracy | Recall | FPR | Precision | F1 | TP / TN / FP / FN |
|---|---:|---:|---:|---:|---:|---:|
| Validación | 82,0375 % | 94,1915 % | 30,1166 % | 75,7726 % | 83,9840 % | 480.002 / 356.127 / 153.475 / 29.600 |
| **Test (held-out)** | **81,9831 %** | **94,1362 %** | **30,1701 %** | **75,7292 %** | **83,9354 %** | 479.719 / 355.854 / 153.747 / 29.882 |

Top 5 cuantiles en validación:

| q | Umbral | Accuracy | Recall | FPR | Precision | F1 |
|---:|---:|---:|---:|---:|---:|---:|
| **0,70** | **1,222079** | **82,0375 %** | **94,1915 %** | **30,1166 %** | **75,7726 %** | **83,9840 %** |
| 0,60 | 1,114059 | 77,5120 % | 94,8982 % | 39,8743 % | 70,4136 % | 80,8427 % |
| 0,50 | 1,051132 | 72,5081 % | 95,8240 % | 50,8079 % | 65,3500 % | 77,7061 % |
| 0,80 | 1,423173 | 55,6097 % | 31,3604 % | 20,1410 % | 60,8923 % | 41,3994 % |
| 0,85 | 1,468968 | 54,4436 % | 24,7689 % | 15,8818 % | 60,9311 % | 35,2205 % |

Tiempos: entrenamiento 6,24 s; scoring de validación 72,92 s; búsqueda del
umbral 0,38 s; scoring de test 73,14 s; **0,07 ms/flujo**.

Con el split aleatorio validación y test coinciden (F1 83,9840 % vs 83,9354 %),
lo que confirma que la diferencia anterior era del particionado. El resultado
sigue siendo malo: el modelo no separa bien benignos y anomalías en CIC.

### Variantes sobre el split aleatorio

| Configuración | q | Umbral | Accuracy | Recall | FPR | Precision | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Standard, 100 cortes | 0,70 | 1,222079 | 81,9831 % | 94,1362 % | 30,1701 % | 75,7292 % | 83,9354 % |
| Standard, 200 cortes | 0,70 | 1,200262 | 82,0942 % | 94,1376 % | 29,9491 % | 75,8644 % | 84,0189 % |
| MaxAbs, 100 cortes | 0,60 | −1,017878 | 77,7288 % | 94,5807 % | 39,1232 % | 70,7389 % | 80,9406 % |

Tiempos de las variantes:

| Configuración | Entrenamiento | Scoring de test | Coste por flujo |
|---|---:|---:|---:|
| Standard, 100 cortes | 6,24 s | 73,14 s | 0,07 ms |
| Standard, 200 cortes | 11,95 s | 146,12 s | 0,14 ms |
| MaxAbs, 100 cortes | 6,12 s | 72,52 s | 0,07 ms |

Duplicar los cortes apenas cambia el F1 (83,94 % → 84,02 %) y duplica el coste.
`MaxAbsScaler` empeora el resultado (80,94 %) y dispara el FPR al 39,12 %.

## Comparación con SAD y con UNSW

| Dataset | Split | Modelo | Accuracy | Recall | FPR | Precision | F1 |
|---|---|---|---:|---:|---:|---:|---:|
| NF-CSE-CIC-IDS2018 | Secuencial | LODA | 71,8578 % | 100,0000 % | 56,2844 % | 63,9859 % | 78,0383 % |
| NF-CSE-CIC-IDS2018 | Aleatorio | LODA | 81,9831 % | 94,1362 % | 30,1701 % | 75,7292 % | 83,9354 % |
| NF-CSE-CIC-IDS2018 | Secuencial | SAD | 89,8337 % | 96,7247 % | 17,0573 % | 85,0088 % | 90,4891 % |
| NF-CSE-CIC-IDS2018 | Aleatorio | SAD | 91,1695 % | 89,1384 % | 6,7994 % | 92,9127 % | 90,9864 % |
| NF-UNSW-NB15 | Secuencial | LODA | 94,9175 % | 100,0000 % | 10,1649 % | 90,7730 % | 95,1634 % |
| NF-UNSW-NB15-v2 | Secuencial | LODA | 98,6829 % | 100,0000 % | 2,6343 % | 97,4333 % | 98,7000 % |

Los valores de SAD proceden de `resultados_sad_cic_ids2018.md`; los de UNSW,
de `resultados_loda_unsw.md` y `resultados_loda_unsw_v2.md`. No son ejecuciones
idénticas: distinto modelo, protocolo y número de repeticiones.

## Observaciones

1. **LODA no funciona bien en CIC-IDS2018.** Incluso con el split aleatorio,
   el mejor resultado es ~82 % de accuracy, ~84 % de F1 y ~30 % de FPR. No
   cumple los objetivos del paper (accuracy > 98 %, FPR < 3,1 %, recall 100 %).
2. **El split secuencial engaña.** El test secuencial tiene recall 100 % y F1
   78,04 %, pero el FPR es 56,28 %: el modelo marca como anómalos más de la
   mitad de los benignos. La subida de recall entre validación y test confirma
   el cambio de distribución.
3. **El split aleatorio mejora el F1 respecto al secuencial (78,04 % → 83,94 %)
   y reduce el FPR (56,28 % → 30,17 %), pero sigue siendo malo.** Además es más
   honesto: validación y test coinciden, así que el ~84 % de F1 es una
   estimación estable del rendimiento de LODA en CIC con estas características.
4. **Más cortes no arreglan el problema.** Pasar de 100 a 200 cortes mejora el
   F1 solo 0,08 puntos y duplica el tiempo de scoring. `MaxAbsScaler` es peor
   que `StandardScaler`.
5. **CIC-IDS2018 es un dataset difícil para este tipo de detectores.** SAD
   tampoco cumple los objetivos (F1 ~91 %, FPR ~6,8 % con split aleatorio),
   aunque queda muy por encima de LODA.
6. **La diferencia con UNSW es grande.** LODA pasa de F1 95,16 % en UNSW v1 y
   98,70 % en v2 a 83,94 % en CIC. El buen resultado en UNSW no se generaliza a
   CIC con las mismas características y el mismo entrenamiento.
7. **Una sola ejecución por configuración.** Los resultados corresponden a una
   partición fija (semilla 42); no se reporta media ± desviación típica.
8. **Validación y test están balanceados al 50 %.** El FPR y el recall deben
   interpretarse con esa proporción; en tráfico real, mucho más desbalanceado,
   la precisión sería menor.

## Reproducibilidad

Split secuencial:

```bash
cd "/home/alberto/Online machine learning/codigo-paper-oml-computer-security"
.venv/bin/python loda_unsw.py --filename ../NF-CSE-CIC-IDS2018.csv
```

Split aleatorio (crear antes el CSV barajado con el fragmento de Python de
arriba):

```bash
.venv/bin/python loda_unsw.py \
  --filename /tmp/opencode/NF-CSE-CIC-IDS2018-shuffled.csv
```

Variantes:

```bash
.venv/bin/python loda_unsw.py \
  --filename /tmp/opencode/NF-CSE-CIC-IDS2018-shuffled.csv \
  --n-random-cuts 200

.venv/bin/python loda_unsw.py \
  --filename /tmp/opencode/NF-CSE-CIC-IDS2018-shuffled.csv \
  --scaling-type MaxAbsScaler
```

Dependencias: `river`, `pandas`, `numpy`, `scikit-learn`, `matplotlib` y `pyyaml`
(las tres últimas porque `loda_unsw.py` reutiliza `hyperparameter_optimization.py`,
que las importa a nivel de módulo).

El script fija `np.random.seed(42)` antes de generar las particiones y usa
`seed=42` en LODA. Por defecto no escribe ningún fichero de métricas; si se
quiere una copia JSON, usar `--output-json`.

## Otros datasets

- NF-UNSW-NB15 v1: `resultados_loda_unsw.md` (F1 95,16 %, FPR 10,16 %).
- NF-UNSW-NB15 v2: `resultados_loda_unsw_v2.md` (F1 98,70 %, FPR 2,63 %).
- NF-ToN-IoT: `resultados_loda_ton_iot.md` (mejor F1 84,19 %, FPR 37,56 %).
- NF-UQ-NIDS: `resultados_loda_uq_nids.md` (F1 70,61 %, FPR 50,19 %).
