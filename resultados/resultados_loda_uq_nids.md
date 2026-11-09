# Modelo LODA sobre NF-UQ-NIDS (entrenamiento directo con benignos)

Prueba de `river.anomaly.LODA` sobre NF-UQ-NIDS con el protocolo habitual:
entrenamiento directo con 100.000 flujos benignos, umbral calibrado con
cuantiles de los scores de entrenamiento y evaluación con el detector congelado.

UQ-NIDS es la unión de cuatro datasets (CIC-IDS2018, UNSW-NB15, ToN-IoT y
BoT-IoT) y está ordenado por dataset de origen. Por eso **no se usa el split
secuencial**: se reduce el fichero a las 9 columnas del modelo, se descarta una
fila con IP inválida y se baraja con `random.seed(42)` antes de partirlo.

El resultado es el peor de todos los datasets probados: LODA marca como
anómalos la mitad de los flujos benignos.

## Metadatos

| Campo | Valor |
|---|---|
| Fecha de ejecución | 2026-10-28 |
| Rama | `feature/more-models` |
| Commit | `7d7e8a3` + cambios locales (`loda_detector.py`, documentos de resultados) |
| Script | `loda_detector.py` |
| Ejecuciones | 1 (StandardScaler, 100 cortes, split aleatorio) |
| Semilla | `42` (barajado, split y LODA) |
| Python | 3.12.3 |
| river | 0.26.1 |
| pandas | 3.1.0 |
| numpy | 2.5.4 |
| scikit-learn | 1.9.1 |
| matplotlib | 3.11.2 |
| PyYAML | 6.0.3 |

## Dataset y composición

| Dataset | Flujos totales | Benignos | Anomalías | Tamaño |
|---|---:|---:|---:|---:|
| NF-UQ-NIDS | 11.994.892 | 9.208.047 | 2.786.845 | 1,0 GB |

Composición según la columna `Dataset`:

| Dataset de origen | Flujos |
|---|---:|
| NF-CSE-CIC-IDS2018 | 8.392.400 |
| NF-UNSW-NB15 | 1.623.118 |
| NF-ToN-IoT | 1.379.274 |
| NF-BoT-IoT | 600.100 |
| **Total** | **11.994.892** |

El fichero reducido contiene una fila con IP de destino inválida
(`172.31.69.25,0,188.,0,0,0,0,0,0`, benigna). Se descarta; quedan 11.994.891
flujos válidos.

Particiones resultantes:

| Conjunto | Flujos |
|---|---:|
| Escalador | 1.000 benignos |
| Entrenamiento | 100.000 benignos |
| Validación | 1.393.422 benignos + 1.393.422 anomalías |
| Test | 1.393.423 benignos + 1.393.423 anomalías |

Características (`FEATURES` de `config.yaml`):

```text
IPV4_SRC_ADDR, IPV4_DST_ADDR, L4_DST_PORT, L4_SRC_PORT, PROTOCOL,
IN_BYTES, OUT_BYTES, FLOW_DURATION_MILLISECONDS, Label
```

## Preprocesado

Para reducir memoria se conservan solo las 9 columnas del modelo:

```bash
cut -d, -f1,2,3,4,5,7,8,12,13 "../NF-UQ-NIDS.csv" \
  > /tmp/opencode/NF-UQ-NIDS-reduced.csv
```

Después se validan las dos columnas de IP, se descarta la fila inválida y se
baraja el fichero:

```python
import ipaddress
import random
from pathlib import Path

src = Path('/tmp/opencode/NF-UQ-NIDS-reduced.csv')
dst = Path('/tmp/opencode/NF-UQ-NIDS-reduced-clean-shuffled.csv')

with src.open('r', newline='') as f:
    header = f.readline()
    rows = f.readlines()

valid = []
invalid = 0
for line in rows:
    fields = line.rstrip('\n').split(',')
    if len(fields) < 3:
        invalid += 1
        continue
    try:
        ipaddress.ip_address(fields[0])
        ipaddress.ip_address(fields[2])
    except ValueError:
        invalid += 1
        continue
    valid.append(line)

random.seed(42)
random.shuffle(valid)

with dst.open('w', newline='') as f:
    f.write(header)
    f.writelines(valid)

print('valid', len(valid), 'invalid', invalid)
```

Resultado: `valid 11994891 invalid 1`.

## Configuración

| Parámetro | Valor |
|---|---|
| `FILENAME` | `/tmp/opencode/NF-UQ-NIDS-reduced-clean-shuffled.csv` |
| `TYPE_IP` | `IPv4` |
| `SIGNIFICANT_PORT` | `False` |
| `BENIGN_FLOWS_ADDITIONAL` | `0` |
| `VALIDATION_RATIO` | `0.5` |
| `FLOWS_TRAIN_SCALER` | `1000` |
| `FLOWS_TRAIN_OML` | `100000` |
| `scaling_type` | `StandardScaler` |
| `n_bins` | `10` |
| `n_random_cuts` | `100` |
| `seed` | `42` |
| `SELECTION_METRIC` | `f1` |
| Rejilla de cuantiles | `0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99` |
| Umbral | cuantil `q` de los scores de entrenamiento |
| Evaluación | detector congelado; test evaluado una sola vez |

## Resultados

Cuantil seleccionado en validación: **q = 0,50** (umbral `1,128266`).

| Conjunto | Accuracy | Recall | FPR | Precision | F1 | TP / TN / FP / FN |
|---|---:|---:|---:|---:|---:|---:|
| Validación | 65,8618 % | 81,9293 % | 50,2057 % | 62,0042 % | 70,5876 % | 1.141.621 / 693.845 / 699.577 / 251.801 |
| **Test (held-out)** | **65,8883 %** | **81,9697 %** | **50,1930 %** | **62,0218 %** | **70,6140 %** | 1.142.184 / 694.022 / 699.401 / 251.239 |

Top 5 cuantiles en validación:

| q | Umbral | Accuracy | Recall | FPR | Precision | F1 |
|---:|---:|---:|---:|---:|---:|---:|
| **0,50** | **1,128266** | **65,8618 %** | **81,9293 %** | **50,2057 %** | **62,0042 %** | **70,5876 %** |
| 0,60 | 1,187044 | 64,2116 % | 68,5370 % | 40,1138 % | 63,0801 % | 65,6954 % |
| 0,70 | 1,311899 | 48,1474 % | 26,7227 % | 30,4280 % | 46,7583 % | 34,0090 % |
| 0,80 | 1,461433 | 42,6436 % | 5,5162 % | 20,2290 % | 21,4261 % | 8,7736 % |
| 0,85 | 1,518534 | 44,6014 % | 4,2310 % | 15,0282 % | 21,9688 % | 7,0955 % |

Tiempos: entrenamiento 6,17 s; scoring de validación 212,41 s; búsqueda del
umbral 1,07 s; scoring de test 211,73 s; **0,08 ms/flujo**.

Validación y test coinciden (F1 70,59 % vs 70,61 %), así que el resultado es
estable para este split; el problema no es el particionado, sino la capacidad
del modelo para separar las clases en UQ-NIDS.

## Comparación con otros datasets

| Dataset | Split | Accuracy | Recall | FPR | Precision | F1 |
|---|---|---:|---:|---:|---:|---:|
| NF-UQ-NIDS | Aleatorio | 65,8883 % | 81,9697 % | 50,1930 % | 62,0218 % | 70,6140 % |
| NF-ToN-IoT | Aleatorio (subconjunto) | 81,2191 % | 99,9988 % | 37,5607 % | 72,6949 % | 84,1884 % |
| NF-CSE-CIC-IDS2018 | Aleatorio | 81,9831 % | 94,1362 % | 30,1701 % | 75,7292 % | 83,9354 % |
| NF-UNSW-NB15 | Secuencial | 94,9175 % | 100,0000 % | 10,1649 % | 90,7730 % | 95,1634 % |
| NF-UNSW-NB15-v2 | Secuencial | 98,6829 % | 100,0000 % | 2,6343 % | 97,4333 % | 98,7000 % |

Los valores de ToN, CIC y UNSW proceden de `resultados_loda_ton_iot.md`,
`resultados_loda_cic_ids2018.md`, `resultados_loda_unsw.md` y
`resultados_loda_unsw_v2.md`. No son ejecuciones idénticas: distinto dataset,
split y número de repeticiones.

## Observaciones

1. **UQ-NIDS es el peor dataset para LODA.** Con el split aleatorio se obtiene
   ~65,89 % de accuracy, ~70,61 % de F1, ~50,19 % de FPR y ~81,97 % de recall.
   El modelo marca como anómalos la mitad de los benignos y se deja ~18 % de las
   anomalías.
2. **El problema no es el split.** Validación y test coinciden, y el fichero se
   barajó antes de partir. La baja calidad es del modelo sobre esta mezcla de
   distribuciones.
3. **UQ-NIDS es una mezcla heterogénea.** El 70 % de los flujos son de
   CIC-IDS2018, que ya era difícil para LODA (F1 ~83,94 % con split aleatorio);
   añadir UNSW, ToN-IoT y BoT-IoT no mejora la separación.
4. **El umbral elegido es el mínimo de la rejilla (q = 0,50).** El modelo
   querría bajar aún más el umbral para recuperar anomalías, pero el FPR ya está
   en el 50 %. No hay un punto de operación con recall alto y FPR bajo.
5. **No se probaron variantes en UQ-NIDS por coste.** Cada ejecución tarda ~7
   minutos solo en scoring. En CIC y ToN, ni 200 cortes ni `MaxAbsScaler`
   arreglaron el problema; es poco probable que aquí cambien la conclusión.
6. **Una sola ejecución.** Los resultados corresponden a un split fijo (semilla
   42); no se reporta media ± desviación típica.
7. **Validación y test están balanceados al 50 %.** El FPR y el recall deben
   interpretarse con esa proporción; en tráfico real, mucho más desbalanceado,
   la precisión sería menor.

## Reproducibilidad

```bash
cd "/home/alberto/Online machine learning/codigo-paper-oml-computer-security"

# 1. Reducir columnas
cut -d, -f1,2,3,4,5,7,8,12,13 "../NF-UQ-NIDS.csv" \
  > /tmp/opencode/NF-UQ-NIDS-reduced.csv

# 2. Limpiar IPs inválidas y barajar (ver script de Python arriba)

# 3. Ejecutar LODA
.venv/bin/python loda_detector.py \
  --filename /tmp/opencode/NF-UQ-NIDS-reduced-clean-shuffled.csv
```

Dependencias: `river`, `pandas`, `numpy`, `scikit-learn`, `matplotlib` y `pyyaml`
(las tres últimas porque `loda_detector.py` reutiliza `hyperparameter_optimization.py`,
que las importa a nivel de módulo).

El script fija `np.random.seed(42)` antes de generar las particiones y usa
`seed=42` en LODA. Por defecto no escribe ningún fichero de métricas; si se
quiere una copia JSON, usar `--output-json`.
