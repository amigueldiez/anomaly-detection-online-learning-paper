# LODA sobre NF-CSE-CIC-IDS2018 (protocolo online)

Evaluación de `river.anomaly.LODA` sobre NF-CSE-CIC-IDS2018 v1 con el
protocolo online: el modelo aprende de los flujos benignos de warm-up, el
umbral se selecciona en validación y, durante la evaluación, el detector se
reentrena con cada flujo que no marca como anómalo. El test se evalúa una
sola vez.

Antes de las repeticiones se hizo un barrido de hiperparámetros sobre el
dataset. Este documento sustituye a los resultados anteriores de LODA en CIC,
obtenidos con el detector congelado y ya no comparables.

## Metadatos

| Campo | Valor |
|---|---|
| Fecha de ejecución | 2026-11-10 |
| Rama | `loda-online-evaluation` |
| Commit base | `303f735` |
| Scripts | `loda_detector.py`, `multiexecution_cic.py` |
| Barrido | 45 configuraciones (5 scalers × 3 `n_bins` × 3 `n_random_cuts`), semilla 42 |
| Multisemilla | 12 ejecuciones (semillas 0-11) con la mejor configuración |
| Python | 3.12.3 |
| river | 0.26.1 |
| pandas | 3.1.0 |
| numpy | 2.5.4 |
| scikit-learn | 1.9.1 |

## Dataset y particiones

| Dataset | Flujos totales | Benignos | Anomalías | Tamaño |
|---|---:|---:|---:|---:|
| NF-CSE-CIC-IDS2018 (v1) | 8.392.399 | 7.373.196 | 1.019.203 | 592 MB |

| Conjunto | Flujos |
|---|---:|
| Escalador | 1.000 benignos |
| Entrenamiento (warm-up) | 100.000 benignos |
| Validación | 509.602 benignos + 509.602 anomalías |
| Test | 509.601 benignos + 509.601 anomalías |

Características (`FEATURES` de `config.yaml`):

```text
IPV4_SRC_ADDR, IPV4_DST_ADDR, L4_DST_PORT, L4_SRC_PORT, PROTOCOL,
IN_BYTES, OUT_BYTES, FLOW_DURATION_MILLISECONDS, Label
```

La columna `Attack` se descarta antes de construir las particiones.

## Protocolo

| Parámetro | Valor |
|---|---|
| `FILENAME` | `/tmp/opencode/NF-CSE-CIC-IDS2018-shuffled.csv` |
| Barajado del CSV | `random.seed(42)` |
| `FLOWS_TRAIN_SCALER` | 1.000 |
| `FLOWS_TRAIN_OML` | 100.000 |
| `SELECTION_METRIC` | F1 |
| Rejilla de cuantiles | `0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99` |
| Umbral | cuantil `q` de los scores de entrenamiento |
| Actualización | online con los flujos que no se marcan como anómalos |

El CSV se baraja una sola vez, de modo que la pertenencia de cada flujo a los
conjuntos es fija. Cada semilla cambia el orden interno de los conjuntos y las
proyecciones aleatorias de LODA, así que la desviación típica mide la
sensibilidad al orden de aprendizaje online, no la variabilidad del
particionado.

## Barrido de hiperparámetros

Rejilla completa de 45 configuraciones con la semilla 42:

- 5 scalers: `MaxAbsScaler`, `MinMaxScaler`, `Normalizer`, `RobustScaler`,
  `StandardScaler`.
- 3 valores de `n_bins`: 5, 10, 20.
- 3 valores de `n_random_cuts`: 50, 100, 200.

Mejor configuración por scaler según F1 de validación:

| Scaler | Bins | Cuts | q | F1 val | F1 test | FPR test |
|---|---:|---:|---:|---:|---:|---:|
| **Normalizer** | **5** | **100** | **0.90** | **90.2930 %** | **90.2535 %** | **12.6707 %** |
| Normalizer | 5 | 200 | 0.85 | 88.5799 % | 88.5396 % | 17.1754 % |
| RobustScaler | 5 | 200 | 0.70 | 84.6949 % | 84.6487 % | 29.0357 % |
| StandardScaler | 20 | 50 | 0.70 | 84.2154 % | 84.1552 % | 29.6304 % |
| MinMaxScaler | 10 | 200 | 0.70 | 83.1539 % | 83.1084 % | 30.2474 % |
| MaxAbsScaler | 20 | 200 | 0.60 | 81.8964 % | 81.8377 % | 36.3504 % |

Las siete mejores configuraciones por F1 de validación son todas `Normalizer`,
así que la elección no depende de un empate afortunado. El resto de scalers se
queda en ~82-85 % de F1 con ~29-36 % de FPR.

Configuración seleccionada: **`Normalizer`, `n_bins=5`, `n_random_cuts=100`**.

El barrido completo está en `resultados/barrido_loda_cic/`:

- `resumen_barrido_loda_cic.json`
- `resultados_barrido_loda_cic.md` (las 45 configuraciones)
- `part1/raw/` y `part2/raw/` (JSON por configuración)

## Resultados con 12 semillas

| Split | Accuracy | Recall | FPR | Precision | F1 |
|---|---:|---:|---:|---:|---:|
| Validación | 90.4025 ± 4.2028 % | 91.9228 ± 2.0002 % | 11.1178 ± 9.5557 % | 89.8450 ± 6.5847 % | 90.6951 ± 3.2472 % |
| Test | 90.3442 ± 4.1900 % | 91.8797 ± 1.9985 % | 11.1913 ± 9.5363 % | 89.7759 ± 6.5839 % | 90.6385 ± 3.2366 % |

Umbral seleccionado: q = 0.9167 ± 0.0718. Tiempo de test: 0.0956 ± 0.0037
ms/flujo.

### Resultados por semilla

| Semilla | q | Umbral | F1 val | FPR val | F1 test | FPR test |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.95 | -3.316768 | 91.2950 % | 10.3339 % | 91.2689 % | 10.3346 % |
| 1 | 0.90 | -3.007146 | 90.1240 % | 13.0641 % | 90.0810 % | 13.1004 % |
| 2 | 0.90 | -3.684186 | 90.6557 % | 11.8147 % | 90.6317 % | 11.8028 % |
| 3 | 0.95 | -2.705487 | 93.4056 % | 5.4735 % | 93.3990 % | 5.4490 % |
| 4 | 0.95 | -3.677108 | 90.9584 % | 6.1809 % | 90.9846 % | 6.1454 % |
| 5 | 0.95 | -3.292766 | 92.7901 % | 6.9230 % | 92.7732 % | 6.9107 % |
| 6 | 0.95 | -3.189214 | 89.9223 % | 9.9356 % | 89.9438 % | 9.6008 % |
| 7 | 0.95 | -2.808453 | 92.0737 % | 8.5531 % | 92.0511 % | 8.5626 % |
| 8 | 0.70 | -3.524685 | 80.9410 % | 40.2944 % | 80.9223 % | 40.2364 % |
| 9 | 0.95 | -2.233785 | 91.8916 % | 3.9929 % | 91.9384 % | 3.9180 % |
| 10 | 0.90 | -3.151870 | 91.7974 % | 9.1897 % | 91.2020 % | 10.5804 % |
| 11 | 0.95 | -3.202862 | 92.4863 % | 7.6577 % | 92.4658 % | 7.6544 % |

### La semilla 8 es un caso atípico

Once de las doce semillas eligen q ≥ 0,90 y quedan en un rango estrecho:

- F1 de test: 89,94-93,40 %.
- FPR de test: 3,92-13,10 %.

La semilla 8 elige q = 0,70 en validación y se desvía por completo: F1 80,92 %,
FPR 40,24 %. Excluyéndola, el resto de semillas da F1 **91,52 ± 1,11 %** y FPR
**8,55 ± 2,83 %**.

El modelo es estable; lo que varía es qué cuantil gana la selección en
validación. Es el mismo patrón observado en NF-UNSW-NB15-v2. Fijar q en la zona
0,90-0,95 reduciría buena parte de la dispersión, pero se deja como análisis
pendiente para no cambiar el protocolo principal.

## Comparación con el protocolo anterior

Los resultados públicos anteriores de LODA en CIC (F1 ≈ 83,94 %,
FPR ≈ 30,17 %) se obtuvieron con el detector congelado en validación y la
configuración `StandardScaler`, `n_bins=10`, `n_random_cuts=100`. Con el
protocolo online y `Normalizer`, `n_bins=5`, `n_random_cuts=100` el F1 sube a
90,64 % y el FPR baja a 11,19 % de media, a costa de una mayor dispersión.

## Ficheros

- `resultados/multisemilla_loda_cic/`: resultados definitivos.
  - `raw/`: un JSON por semilla.
  - `resumen_multisemilla_loda_cic.json`: resumen agregado completo.
  - `resultados_multisemilla_loda_cic.md`: tablas de este documento.
- `resultados/barrido_loda_cic/`: barrido de hiperparámetros.
- `loda_detector.py`: implementación del protocolo online.
- `multiexecution_cic.py`: driver multisemilla; LODA usa `Normalizer`, 5, 100.
