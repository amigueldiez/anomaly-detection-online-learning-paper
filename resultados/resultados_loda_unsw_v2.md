# Modelo LODA sobre NF-UNSW-NB15-v2 (entrenamiento directo con benignos)

Prueba funcional de `river.anomaly.LODA` sobre NF-UNSW-NB15-v2, con el mismo
protocolo que la prueba de NF-UNSW-NB15 v1 (`resultados_loda_unsw.md`):
entrenamiento directo con flujos benignos, umbral calibrado con cuantiles de los
scores de entrenamiento y evaluación con el detector congelado.

## Metadatos

| Campo | Valor |
|---|---|
| Fecha de ejecución | 2026-10-27 |
| Rama | `feature/more-models` |
| Commit | `7d7e8a3` + cambios locales (`loda_detector.py`, documentos de resultados) |
| Script | `loda_detector.py` |
| Ejecuciones | 1 (configuración principal) |
| Semilla | `42` (split y LODA) |
| Python | 3.12.3 |
| river | 0.26.1 |
| pandas | 3.1.0 |
| numpy | 2.5.4 |
| scikit-learn | 1.9.1 |
| matplotlib | 3.11.2 |
| PyYAML | 6.0.3 |

## Dataset y particiones

| Dataset | Flujos totales | Benignos | Anomalías | Escalador | Entrenamiento | Validación (benignos/anomalías) | Test (benignos/anomalías) |
|---|---:|---:|---:|---:|---:|---:|---:|
| NF-UNSW-NB15-v2 | 2.390.275 | 2.295.222 | 95.053 | 1.000 | 100.000 | 47.526 / 47.526 | 47.527 / 47.527 |

Características (`FEATURES` de `config.yaml`):

```text
IPV4_SRC_ADDR, IPV4_DST_ADDR, L4_DST_PORT, L4_SRC_PORT, PROTOCOL,
IN_BYTES, OUT_BYTES, FLOW_DURATION_MILLISECONDS, Label
```

## Configuración

| Parámetro | Valor |
|---|---|
| `FILENAME` | `../NF-UNSW-NB15-v2.csv` |
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

El protocolo es el mismo que en v1: LODA se entrena solo con los 100.000 flujos
benignos, no se actualiza durante validación ni test, y el umbral se elige en
validación sin tocar test.

## Resultados

Cuantil seleccionado en validación: **q = 0,95** (umbral `1,861260`).

| Conjunto | Accuracy | Recall | FPR | Precision | F1 | TP / TN / FP / FN |
|---|---:|---:|---:|---:|---:|---:|
| Validación | 98,3904 % | 100,0000 % | 3,2193 % | 96,8811 % | 98,4159 % | 47.526 / 45.996 / 1.530 / 0 |
| **Test (held-out)** | **98,6829 %** | **100,0000 %** | **2,6343 %** | **97,4333 %** | **98,7000 %** | 47.527 / 46.275 / 1.252 / 0 |

Los cinco mejores cuantiles en validación fueron:

| q | Umbral | Accuracy | Recall | FPR | Precision | F1 |
|---:|---:|---:|---:|---:|---:|---:|
| **0,95** | **1,861260** | **98,3904 %** | **100,0000 %** | **3,2193 %** | **96,8811 %** | **98,4159 %** |
| 0,90 | 1,677644 | 96,2705 % | 100,0000 % | 7,4591 % | 93,0587 % | 96,4046 % |
| 0,85 | 1,612350 | 94,0906 % | 100,0000 % | 11,8188 % | 89,4304 % | 94,4203 % |
| 0,80 | 1,569013 | 92,4547 % | 100,0000 % | 15,0907 % | 86,8880 % | 92,9840 % |
| 0,70 | 1,519147 | 87,4406 % | 100,0000 % | 25,1189 % | 79,9240 % | 88,8419 % |

Tiempos:

| Fase | Tiempo |
|---|---:|
| Entrenamiento (100.000 flujos) | 5,99 s |
| Scoring del entrenamiento (100.000) | 7,26 s |
| Scoring de validación (95.052) | 6,86 s |
| Búsqueda del umbral (8 cuantiles) | 0,04 s |
| Scoring de test (95.054) | 6,91 s |
| Coste por flujo en test | 0,07 ms/flujo |

## Comparación con v1 y con otros modelos

| Dataset | Modelo | Protocolo | Accuracy | Recall | FPR | Precision | F1 |
|---|---|---|---:|---:|---:|---:|---:|
| NF-UNSW-NB15 | LODA | 100 cortes, 1 ejecución | 94,9175 % | 100,0000 % | 10,1649 % | 90,7730 % | 95,1634 % |
| **NF-UNSW-NB15-v2** | **LODA** | **100 cortes, 1 ejecución** | **98,6829 %** | **100,0000 %** | **2,6343 %** | **97,4333 %** | **98,7000 %** |
| NF-UNSW-NB15-v2 | One-Class SVM | `hyperparameter_optimization.py`, media de 12 ejecuciones | 99,7514 % | 99,5210 % | 0,0182 % | 99,9817 % | 99,7491 % |
| NF-UNSW-NB15-v2 | SAD | `sad_detector.py`, 1 ejecución, umbral 5,7σ | 99,8874 % | 100,0000 % | 0,2251 % | 99,7754 % | 99,8876 % |

Los valores de One-Class SVM proceden de
`resultados_split_entrenamiento_validacion_test.md`; los de SAD, de
`resultados_sad_unsw.md`. No son ejecuciones idénticas: distinto protocolo,
split y número de repeticiones, por lo que la comparación es orientativa.

## Observaciones

1. **v2 mejora claramente a v1 para LODA.** El F1 de test sube de 95,1634 % a
   98,7000 %, el FPR baja de 10,1649 % a 2,6343 % y el recall se mantiene en
   100 %. La separación entre benignos y anomalías es mejor en v2.
2. **Con v2 se cumple el objetivo del paper.** Accuracy > 98 %, FPR < 3,1 % y
   recall 100 % en test: 98,6829 %, 2,6343 % y 100,0000 %.
3. **El cuantil óptimo sube a 0,95** (frente a 0,90 en v1). Al separar mejor las
   clases, se puede exigir un umbral más alto sin perder recall; aun así, bajar
   a q = 0,90 dispara el FPR al 7,46 %.
4. **LODA sigue por debajo de One-Class SVM y SAD en v2.** One-Class SVM alcanza
   ~99,75 % de F1 con FPR ~0,02 %, y SAD ~99,89 % de F1 con FPR ~0,23 %. LODA
   cumple el objetivo, pero no es el mejor detector de los probados.
5. **El coste es similar al de v1:** ~6 s de entrenamiento y ~0,07 ms por flujo
   en test, con 100 cortes.
6. **Una sola ejecución.** Los resultados corresponden a una partición fija
   (semilla 42); no se reporta media ± desviación típica.
7. **Validación y test están balanceados al 50 %.** El FPR y el recall deben
   interpretarse con esa proporción; en tráfico real, mucho más desbalanceado,
   la precisión sería menor.

## Reproducibilidad

```bash
cd "/home/alberto/Online machine learning/codigo-paper-oml-computer-security"
.venv/bin/python loda_detector.py --filename ../NF-UNSW-NB15-v2.csv
```

Dependencias: `river`, `pandas`, `numpy`, `scikit-learn`, `matplotlib` y `pyyaml`
(las tres últimas porque `loda_detector.py` reutiliza `hyperparameter_optimization.py`,
que las importa a nivel de módulo).

El script fija `np.random.seed(42)` antes de generar las particiones y usa
`seed=42` en LODA. Por defecto no escribe ningún fichero de métricas; si se
quiere una copia JSON, usar `--output-json`.

## Otros datasets

- NF-UNSW-NB15 v1: `resultados_loda_unsw.md`.
- NF-CSE-CIC-IDS2018: `resultados_loda_cic_ids2018.md`. Resultado negativo:
  LODA se queda en ~83,94 % de F1 y ~30,17 % de FPR con split aleatorio.
- NF-ToN-IoT: `resultados_loda_ton_iot.md`. Mejor resultado ~84,19 % de F1 y
  ~37,56 % de FPR con `MaxAbsScaler`.
- NF-UQ-NIDS: `resultados_loda_uq_nids.md`. Resultado negativo: ~70,61 % de F1
  y ~50,19 % de FPR con split aleatorio.

## Repeticiones multisemilla en UNSW

Los resultados de este documento corresponden a una sola ejecución. La
evaluación con 12 semillas sobre UNSW v1 y v2 está en
`resultados_multisemilla_sad_loda_unsw.md`. Resumen de test (media ± desviación
típica, n = 12):

| Dataset | Accuracy | Recall | FPR | F1 |
|---|---:|---:|---:|---:|
| NF-UNSW-NB15 | 95,0337 ± 0,4223 % | 99,6962 ± 1,0525 % | 9,6288 ± 1,3981 % | 95,2561 ± 0,3819 % |
| NF-UNSW-NB15-v2 | 98,8770 ± 0,3137 % | 99,9998 ± 0,0006 % | 2,2459 ± 0,6279 % | 98,8903 ± 0,3091 % |
