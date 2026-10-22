# Resultados y parámetros: split entrenamiento / validación / test

Resultados de las ejecuciones del pipeline `hyperparameter_optimization.py` con el
split disjunto scaler / entrenamiento / validación / test.

## Metadatos

| Campo | Valor |
|---|---|
| Fecha de ejecución | 2026-10-22 |
| Rama | `feature/train-validation-test-split` |
| Commit | `0b97074` (*Add paper/ directory to .gitignore*) |
| Script | `hyperparameter_optimization.py` |
| Ejecuciones | 12 por dataset (procesos independientes) |
| Semilla | No fijada; cada ejecución baraja el orden de los flujos |
| Python | 3.12.3 |
| river | 0.21.1 |
| pandas | 2.2.2 |
| scikit-learn | 1.5.0 |

## Dataset

| Dataset | Flujos totales | Benignos | Anomalías |
|---|---:|---:|---:|
| NF-UNSW-NB15 | 1.623.118 | 1.550.712 | 72.406 |
| NF-UNSW-NB15-v2 | 2.390.275 | 2.295.222 | 95.053 |

## Parámetros

Parámetros comunes:

| Parámetro | Valor |
|---|---|
| `TYPE_IP` | `IPv4` |
| `SIGNIFICANT_PORT` | `False` |
| `BENIGN_FLOWS_ADDITIONAL` | `0` |
| `scaling_type` | `MaxAbsScaler` |
| `nu_value` (ν) | `0.05` |
| `q_value` (q) | `0.99` |
| `FLOWS_TRAIN_SCALER` | `1000` |
| `FLOWS_TRAIN_OML` | `100000` |
| `VALIDATION_RATIO` | `0.5` |
| `SELECTION_METRIC` | `f1` |
| `logs_save` | `[]` (sin ficheros de log durante estas ejecuciones) |

Learning rate por dataset:

| Dataset | `learning_rate` |
|---|---:|
| NF-UNSW-NB15 | `0.1` |
| NF-UNSW-NB15-v2 | `0.3` |

Características utilizadas (`FEATURES`):

```text
IPV4_SRC_ADDR, IPV4_DST_ADDR, L4_DST_PORT, L4_SRC_PORT, PROTOCOL,
IN_BYTES, OUT_BYTES, FLOW_DURATION_MILLISECONDS, Label
```

Ficheros de datos:

- `NF-UNSW-NB15`: `/home/alberto/Online machine learning/NF-UNSW-NB15.csv`
- `NF-UNSW-NB15-v2`: `/home/alberto/Online machine learning/NF-UNSW-NB15-v2.csv`

## Particiones

Los cuatro conjuntos son disjuntos. El escalador se ajusta solo con el conjunto
de escalado y se actualiza durante el entrenamiento; validación y test solo se
transforman. Validación y test están balanceados al 50 % (mismo número de
benignos y anomalías).

| Dataset | Escalador | Entrenamiento | Validación (benignos/anomalías) | Test (benignos/anomalías) |
|---|---:|---:|---:|---:|
| NF-UNSW-NB15 | 1.000 | 100.000 | 36.203 / 36.203 | 36.203 / 36.203 |
| NF-UNSW-NB15-v2 | 1.000 | 100.000 | 47.526 / 47.526 | 47.527 / 47.527 |

## Resultados agregados (media ± desviación típica, n = 12)

### NF-UNSW-NB15

Configuración: `MaxAbsScaler`, ν = 0.05, q = 0.99, learning rate = `0.1`.

| Métrica | Validación | Test |
|---|---:|---:|
| Accuracy | 98.4114 ± 0.1064 % | 97.7703 ± 0.1057 % |
| Recall | 99.4413 ± 0.2182 % | 99.5007 ± 0.2249 % |
| FPR | 2.6186 ± 0.0077 % | 3.9601 ± 0.0140 % |
| Precision | 97.4343 ± 0.0052 % | 96.1724 ± 0.0053 % |
| F1 | 98.4275 ± 0.1070 % | 97.8081 ± 0.1064 % |

### NF-UNSW-NB15-v2

Configuración: `MaxAbsScaler`, ν = 0.05, q = 0.99, learning rate = `0.3`.

| Métrica | Validación | Test |
|---|---:|---:|
| Accuracy | 99.8211 ± 0.3329 % | 99.7514 ± 0.4200 % |
| Recall | 99.6544 ± 0.6710 % | 99.5210 ± 0.8463 % |
| FPR | 0.0123 ± 0.0073 % | 0.0182 ± 0.0082 % |
| Precision | 99.9877 ± 0.0073 % | 99.9817 ± 0.0082 % |
| F1 | 99.8197 ± 0.3362 % | 99.7491 ± 0.4254 % |

### Tiempos de ejecución

| Dataset | Entrenamiento (100.000 flujos) | Test | µs/flujo en test |
|---|---:|---:|---:|
| NF-UNSW-NB15 | 1.0249 ± 0.0211 s | 0.5299 ± 0.0123 s | 7.32 |
| NF-UNSW-NB15-v2 | 1.0213 ± 0.0149 s | 0.7045 ± 0.0140 s | 7.41 |

## Resultados por ejecución (test)

### NF-UNSW-NB15

| Ejecución | Accuracy (%) | Recall (%) | FPR (%) | Precision (%) | F1 (%) |
|---:|---:|---:|---:|---:|---:|
| 1 | 97.8980 | 99.7680 | 3.9720 | 96.1712 | 97.9366 |
| 2 | 97.8040 | 99.5774 | 3.9693 | 96.1667 | 97.8423 |
| 3 | 97.7088 | 99.3675 | 3.9499 | 96.1769 | 97.7461 |
| 4 | 97.7737 | 99.5111 | 3.9638 | 96.1693 | 97.8117 |
| 5 | 97.6148 | 99.1631 | 3.9334 | 96.1848 | 97.6512 |
| 6 | 97.7281 | 99.4089 | 3.9527 | 96.1758 | 97.7656 |
| 7 | 97.6604 | 99.2680 | 3.9472 | 96.1758 | 97.6974 |
| 8 | 97.9035 | 99.7845 | 3.9776 | 96.1666 | 97.9422 |
| 9 | 97.9684 | 99.9199 | 3.9831 | 96.1665 | 98.0073 |
| 10 | 97.7005 | 99.3564 | 3.9555 | 96.1713 | 97.7379 |
| 11 | 97.7336 | 99.4255 | 3.9582 | 96.1713 | 97.7713 |
| 12 | 97.7502 | 99.4586 | 3.9582 | 96.1725 | 97.7880 |

### NF-UNSW-NB15-v2

| Ejecución | Accuracy (%) | Recall (%) | FPR (%) | Precision (%) | F1 (%) |
|---:|---:|---:|---:|---:|---:|
| 1 | 99.3961 | 98.8028 | 0.0105 | 99.9894 | 99.3925 |
| 2 | 99.9695 | 99.9663 | 0.0274 | 99.9726 | 99.9695 |
| 3 | 99.9821 | 99.9937 | 0.0295 | 99.9705 | 99.9821 |
| 4 | 99.7843 | 99.5834 | 0.0147 | 99.9852 | 99.7839 |
| 5 | 99.9558 | 99.9348 | 0.0231 | 99.9768 | 99.9558 |
| 6 | 99.9674 | 99.9558 | 0.0210 | 99.9790 | 99.9674 |
| 7 | 98.5955 | 97.1953 | 0.0042 | 99.9957 | 98.5756 |
| 8 | 99.4603 | 98.9290 | 0.0084 | 99.9915 | 99.4574 |
| 9 | 99.9663 | 99.9495 | 0.0168 | 99.9832 | 99.9663 |
| 10 | 99.9811 | 99.9895 | 0.0274 | 99.9727 | 99.9811 |
| 11 | 99.9884 | 100.0000 | 0.0231 | 99.9769 | 99.9884 |
| 12 | 99.9695 | 99.9516 | 0.0126 | 99.9874 | 99.9695 |

## Observaciones

1. **NF-UNSW-NB15: el test es peor que la validación.** La accuracy baja de
   98.4114 % a 97.7703 % y el FPR sube de 2.6186 % a 3.9601 %. La desviación
   entre ejecuciones es muy pequeña (0.0140 puntos de FPR), por lo que la
   diferencia es sistemática y no ruido de muestreo. Los benignos de validación
   y test son bloques consecutivos del CSV (filas aproximadas 109k–148k y
   148k–186k), mientras que las anomalías se reparten por todo el fichero, lo
   que introduce una deriva entre ambos bloques.

2. **Las 12 ejecuciones no remuestrean la partición.** `split_flows` fija la
   pertenencia de cada flujo a scaler/entrenamiento/validación/test y solo
   baraja el orden dentro de cada conjunto. Por tanto, la desviación típica
   refleja el efecto del orden del aprendizaje online, no la variabilidad de
   la partición. Para errores de barra representativos del split habría que
   repetir con particiones aleatorias distintas.

3. **NF-UNSW-NB15-v2 es casi perfecto de media pero inestable.** En 4 de las
   12 ejecuciones el recall de test cae a 97.20–99.58 % (umbral demasiado alto,
   FPR próximo a 0), mientras que en las otras 8 ronda el 99.9 %. De ahí la
   desviación de 0.8463 puntos en recall. Un umbral congelado o calibrado sobre
   benignos de validación podría estabilizar este comportamiento.

4. **Implicaciones para el abstract.** Para v2 se cumple accuracy > 98 % y
   FPR < 3.1 %, pero el recall medio es 99.5210 %, no 100 % (solo 1 de las 12
   ejecuciones dio exactamente 100 %). Para v1, con el nuevo split, la
   accuracy de test es 97.7703 % y el FPR 3.9601 %, por lo que las cifras
   globales del abstract deben matizarse.

5. **Selección de hiperparámetros.** Solo se evaluó una combinación por
   dataset, por lo que la fase de selección sobre validación es trivial. Para
   confirmar que estas configuraciones siguen siendo las mejores con el nuevo
   split hay que ejecutar la rejilla completa.

## Reproducibilidad

Cada ejecución se lanzó como un proceso nuevo con el `config.yaml`
correspondiente (una combinación por dataset):

```bash
python3 hyperparameter_optimization.py
```

Los logs resumidos de las 12 ejecuciones utilizados para este documento están en:

- `/tmp/oml_runs/nf15/summary.log`
- `/tmp/oml_runs/nf15v2/summary.log`
