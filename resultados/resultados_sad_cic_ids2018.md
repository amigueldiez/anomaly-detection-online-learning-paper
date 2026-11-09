# Modelo StandardAbsoluteDeviation (SAD) sobre CIC-IDS2018

Prueba funcional del detector `river.anomaly.StandardAbsoluteDeviation` sobre el
dataset NF-CSE-CIC-IDS2018 (v1), con el mismo esquema de particiones y el mismo
protocolo de selección de umbral que la prueba de NF-UNSW-NB15
(`resultados_sad_unsw.md`).

## Metadatos

| Campo | Valor |
|---|---|
| Fecha de ejecución | 2026-10-26 |
| Rama | `feature/more-models` |
| Commit | `7857f76` + cambios locales en `sad_detector.py` (`--threshold-step`) |
| Script | `sad_detector.py` |
| Ejecuciones | 1 por protocolo de partición |
| Python | 3.12.3 |
| river | 0.26.1 |
| pandas | 3.1.0 |
| numpy | 2.5.4 |
| scikit-learn | 1.9.1 |
| matplotlib | 3.11.2 |
| PyYAML | 6.0.3 |

## Dataset

| Dataset | Flujos totales | Benignos | Anomalías | Tamaño |
|---|---:|---:|---:|---:|
| NF-CSE-CIC-IDS2018 (v1) | 8.392.399 | 7.373.196 | 1.019.203 | 592 MB |
| NF-CSE-CIC-IDS2018-v2 | 18.893.708 | 16.635.567 | 2.258.141 | 3,2 GB |

En este documento se evalúa **v1**. El v2 no se ejecutó por su tamaño (18,9
millones de flujos y 3,2 GB); queda pendiente si se quiere completar la
comparación.

## Modelo y protocolo

Se reutiliza `sad_detector.py` con esta configuración:

| Parámetro | Valor |
|---|---|
| `FILENAME` | `../NF-CSE-CIC-IDS2018.csv` (o el CSV barajado para el split aleatorio) |
| `TYPE_IP` | `IPv4` |
| `SIGNIFICANT_PORT` | `False` |
| `BENIGN_FLOWS_ADDITIONAL` | `0` |
| `VALIDATION_RATIO` | `0.5` |
| `scaling_type` | `MaxAbsScaler` |
| Modelo | `StandardAbsoluteDeviation(sub_stat=stats.Mean(), varianza=stats.Var)` |
| Agregación | `max` |
| `FLOWS_TRAIN_SCALER` | `1000` |
| `FLOWS_TRAIN_OML` | `100000` |
| `SELECTION_METRIC` | `f1` |
| Rejilla de umbrales | `0,0–10,0σ` paso `0,5` + `12/15/20` (24 candidatos) |
| Semilla del split | `42` |
| Filtro de varianza | `> 1e-12` |
| Actualización online | solo si el flujo **no** se marca como anómalo |

Características utilizadas (`FEATURES` de `config.yaml`):

```text
IPV4_SRC_ADDR, IPV4_DST_ADDR, L4_DST_PORT, L4_SRC_PORT, PROTOCOL,
IN_BYTES, OUT_BYTES, FLOW_DURATION_MILLISECONDS, Label
```

Se usa un paso de 0,5σ en vez del 0,1σ de UNSW porque cada pasada de validación
recorre ~1.019.000 flujos y la búsqueda completa tardaría ~10 minutos por
ejecución.

Particiones:

| Conjunto | Flujos |
|---|---:|
| Escalador | 1.000 benignos |
| Entrenamiento | 100.000 benignos |
| Validación | 509.602 benignos + 509.602 anomalías |
| Test | 509.601 benignos + 509.601 anomalías |

## Dos formas de partir el dataset

El pipeline principal (`generate_dataset`) construye validación y test tomando
**bloques consecutivos** del CSV: los benignos de validación son las filas
siguientes a las de entrenamiento y los de test, el bloque siguiente; las
anomalías se reparten por orden de aparición. Eso funciona si el fichero está
bien mezclado, pero en CIC-IDS2018 los flujos están agrupados por capturas y
tipos de ataque, así que validación y test pueden acabar con distribuciones
distintas.

Para separar el efecto del modelo del efecto del split se han hecho dos
experimentos:

1. **Split secuencial** (comportamiento por defecto del pipeline).
2. **Split aleatorio**: se baraja el CSV completo con `random.seed(42)` y se
   aplica el mismo `generate_dataset` sobre el fichero barajado. Así validación
   y test son muestras aleatorias de todo el dataset.

## Resultados

### Split secuencial (por defecto)

Umbral seleccionado en validación: **3,5σ**.

| Conjunto | Accuracy | Recall | FPR | Precision | F1 | TP / TN / FP / FN |
|---|---:|---:|---:|---:|---:|---:|
| Validación | 83,4779 % | 84,7412 % | 17,7854 % | 82,6529 % | 83,6840 % | 431.843 / 418.967 / 90.635 / 77.759 |
| Test | 89,8337 % | 96,7247 % | 17,0573 % | 85,0088 % | 90,4891 % | 492.910 / 422.677 / 86.924 / 16.691 |

La diferencia entre validación y test (el recall sube 12 puntos y el F1 sube
6,8 puntos) indica que los dos bloques no representan la misma distribución.

### Split aleatorio (reproducible, `random.seed(42)`)

Umbral seleccionado en validación: **3,0σ**.

| Conjunto | Accuracy | Recall | FPR | Precision | F1 | TP / TN / FP / FN |
|---|---:|---:|---:|---:|---:|---:|
| Validación | 91,0943 % | 89,0621 % | 6,8734 % | 92,8354 % | 90,9096 % | 453.862 / 474.575 / 35.027 / 55.740 |
| **Test (held-out)** | **91,1695 %** | **89,1384 %** | **6,7994 %** | **92,9127 %** | **90,9864 %** | 454.250 / 474.951 / 34.650 / 55.351 |

Con el split aleatorio validación y test coinciden, lo que confirma que el
modelo es estable y que la diferencia anterior era del particionado. El
resultado, aun así, es mucho peor que en UNSW.

### Tiempos

| Split | Búsqueda del umbral | Entrenamiento final | Test |
|---|---:|---:|---:|
| Secuencial | 144,72 s (24 candidatos) | 0,19 s | 5,20 s |
| Aleatorio | 144,66 s (24 candidatos) | 0,18 s | 5,25 s |

La ejecución secuencial completa tardó 3 min 19 s y usó un máximo de ~3,3 GB de
RAM.

## Comparación con UNSW

Resultados de test (cada dataset con su protocolo documentado; en UNSW el split
es el secuencial del pipeline, en CIC se usa el aleatorio):

| Dataset | Umbral | Accuracy | Recall | FPR | F1 |
|---|---:|---:|---:|---:|---:|
| NF-UNSW-NB15 | 4,9σ | 97,2143 % | 100,0000 % | 5,5714 % | 97,2898 % |
| NF-UNSW-NB15-v2 | 5,7σ | 99,8874 % | 100,0000 % | 0,2251 % | 99,8876 % |
| NF-CSE-CIC-IDS2018 v1 | 3,0σ | 91,1695 % | 89,1384 % | 6,7994 % | 90,9864 % |

## Observaciones

1. **SAD funciona, pero no tan bien como en UNSW.** En CIC-IDS2018 v1 el mejor
   resultado reproducible es ~91 % de accuracy y ~91 % de F1, con ~6,8 % de FPR
   y ~89 % de recall. No cumple los objetivos del paper (accuracy > 98 %,
   FPR < 3,1 %, recall 100 %).
2. **El split secuencial del pipeline distorsiona la evaluación.** Con bloques
   consecutivos, validación y test no son comparables (F1 83,7 % frente a
   90,5 %). Barajando el fichero, ambas coinciden (~91 %), lo que da una medida
   más fiable del modelo.
3. **La suposición gaussiana univariante es limitada en CIC.** El tráfico
   benigno de CIC-IDS2018 es mucho más heterogéneo (servicios, capturas y días
   distintos) y hay valores extremos muy grandes (`IN_BYTES` llega a 24 millones).
   Una media y una varianza globales por característica no separan bien todas
   las subpoblaciones benignas, y `MaxAbsScaler` comprime la mayoría de valores
   hacia cero.
4. **El umbral óptimo es más bajo que en UNSW** (3,0σ frente a 4,9–5,7σ) y el
   F1 cae rápido al bajarlo: a 2,5σ el recall sube al 92,5 % pero el FPR se
   dispara al 21,8 %. No hay un punto de operación con recall alto y FPR bajo.
5. **El split aleatorio es reproducible.** El fichero se barajó con
   `random.seed(42)`; el `--seed 42` del script controla además el orden dentro
   de cada conjunto.
6. **Una sola ejecución por split.** No se reporta media ± desviación típica.
   Para ello habría que repetir con varias semillas de barajado del fichero.

## Reproducibilidad

Split secuencial:

```bash
cd "/home/alberto/Online machine learning/codigo-paper-oml-computer-security"
.venv/bin/python sad_detector.py \
  --filename ../NF-CSE-CIC-IDS2018.csv \
  --threshold-step 0.5
```

Split aleatorio:

```bash
cd "/home/alberto/Online machine learning"
python3 - <<'EOF'
import random
random.seed(42)
with open('NF-CSE-CIC-IDS2018.csv', newline='') as f:
    header = f.readline()
    lines = f.readlines()
random.shuffle(lines)
with open('/tmp/opencode/cic_shuffled_py42.csv', 'w', newline='') as f:
    f.write(header)
    f.writelines(lines)
EOF

cd codigo-paper-oml-computer-security
.venv/bin/python sad_detector.py \
  --filename /tmp/opencode/cic_shuffled_py42.csv \
  --threshold-step 0.5
```

El resto de opciones se dejan en su valor por defecto, documentado en la tabla
de parámetros.

## Repeticiones multisemilla en CIC

Los resultados de este documento corresponden a una sola ejecución. La
evaluación con 12 semillas sobre el mismo CSV barajado (partición fija, solo
cambia el orden interno de los conjuntos) está en
`resultados_multisemilla_cic_ids2018.md`. Resumen de test (media ± desviación
típica, n = 12):

| Dataset | Accuracy | Recall | FPR | F1 |
|---|---:|---:|---:|---:|
| NF-CSE-CIC-IDS2018 | 91,1956 ± 0,0258 % | 89,1866 ± 0,0470 % | 6,7954 ± 0,0427 % | 91,0151 ± 0,0269 % |

El umbral seleccionado fue 3,0σ en las 12 ejecuciones y la desviación es muy
pequeña: la ejecución individual es representativa del rendimiento medio.
