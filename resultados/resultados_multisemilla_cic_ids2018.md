# Repeticiones multisemilla de SAD, LODA y One-Class SVM sobre NF-CSE-CIC-IDS2018

Evaluación de `river.anomaly.StandardAbsoluteDeviation` (SAD),
`river.anomaly.LODA` y el pipeline One-Class SVM de
`hyperparameter_optimization.py` con **12 ejecuciones por modelo** sobre
NF-CSE-CIC-IDS2018 v1.

El CSV se barajó una sola vez con `random.seed(42)`, de modo que la pertenencia
de cada flujo a los conjuntos de escalador / entrenamiento / validación / test es
fija. Cada semilla cambia únicamente el orden interno de esos conjuntos (y, en
LODA, también las proyecciones aleatorias del modelo). Por tanto, la desviación
típica mide la sensibilidad al orden de aprendizaje online, **no** la
variabilidad del particionado. Para medir esta última habría que barajar el CSV
con una semilla distinta en cada ejecución.

## Metadatos

| Campo | Valor |
|---|---|
| Fecha de ejecución | 2026-11-06 |
| Rama | `feature/more-models` |
| Commit | `b4dc5ec` + cambios locales (`multiexecution_cic.py`, `ocsvm_cic.py`, `merge_cic_summaries.py` y este documento) |
| Scripts | `multiexecution_cic.py`, `ocsvm_cic.py`, `merge_cic_summaries.py` |
| Ejecuciones | 36 (12 semillas × 3 modelos) |
| Semillas | `0`–`11` |
| Duración total | 2.532,5 s (42 min 12 s); SAD 2.466,7 s, LODA 2.532,5 s, OCSVM 737,1 s |
| Python | 3.12.3 |
| river | 0.26.1 |
| pandas | 3.1.0 |
| numpy | 2.5.4 |
| scikit-learn | 1.9.1 |
| matplotlib | 3.11.2 |
| PyYAML | 6.0.3 |

## Protocolo

Cada ejecución es un proceso independiente que llama al script del modelo con
`--seed <s>` y `--output-json`. El protocolo interno es el ya documentado en las
pruebas individuales:

- **SAD**: un detector por característica, agregación `max`, `MaxAbsScaler`,
  umbral elegido en validación sobre una rejilla de `0,0–10,0σ` con paso `0,5`
  (24 candidatos), modelo reevaluado online y test evaluado una sola vez.
- **LODA**: entrenamiento directo con los 100.000 flujos benignos,
  `StandardScaler`, `n_bins=10`, `n_random_cuts=100`, umbral elegido como
  cuantil de los scores de entrenamiento sobre la rejilla
  `0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99`, detector congelado en validación
  y test.
- **One-Class SVM**: `StandardScaler`, ν = 0.5, q = 0.9, learning rate = 0.3
  (la mejor configuración de la rejilla de 48 combinaciones de
  `resultados_ocsvm_cic_ids2018.md`), con `QuantileFilter`; entrenamiento online
  y actualización durante la evaluación, igual que el pipeline principal.

Parámetros comunes:

| Parámetro | Valor |
|---|---|
| `FILENAME` | `/tmp/opencode/NF-CSE-CIC-IDS2018-shuffled.csv` |
| Barajado del CSV | `random.seed(42)` |
| `FLOWS_TRAIN_SCALER` | 1.000 |
| `FLOWS_TRAIN_OML` | 100.000 |
| `VALIDATION_RATIO` | 0.5 |
| `SELECTION_METRIC` | `f1` |

Particiones (idénticas a las pruebas individuales):

| Dataset | Flujos totales | Benignos | Anomalías | Escalador | Entrenamiento | Validación (benignos/anomalías) | Test (benignos/anomalías) |
|---|---:|---:|---:|---:|---:|---:|---:|
| NF-CSE-CIC-IDS2018 (v1) | 8.392.399 | 7.373.196 | 1.019.203 | 1.000 | 100.000 | 509.602 / 509.602 | 509.601 / 509.601 |

> **Alcance de la desviación típica.** La partición no se remuestrea entre
> semillas: cada flujo pertenece siempre al mismo conjunto. La desviación mide el
> efecto del orden de aprendizaje online (y, en LODA, también el de las
> proyecciones aleatorias). No es una barra de error del particionado.

## Resultados agregados (media ± desviación típica, n = 12)

### SAD

| Parámetro | Conjunto | Accuracy | Recall | FPR | Precision | F1 |
|---|---|---:|---:|---:|---:|---:|
| umbral 3,0000σ ± 0,0000σ | Validación | 91,1127 ± 0,0181 % | 89,0684 ± 0,0545 % | 6,8431 ± 0,0335 % | 92,8652 ± 0,0294 % | 90,9272 ± 0,0211 % |
| | **Test** | **91,1956 ± 0,0258 %** | **89,1866 ± 0,0470 %** | **6,7954 ± 0,0427 %** | **92,9201 ± 0,0403 %** | **91,0151 ± 0,0269 %** |

### LODA

| Parámetro | Conjunto | Accuracy | Recall | FPR | Precision | F1 |
|---|---|---:|---:|---:|---:|---:|
| q 0,8500 ± 0,0953 | Validación | 88,3107 ± 4,4301 % | 91,4095 ± 2,1267 % | 14,7881 ± 9,6279 % | 86,7253 ± 7,2021 % | 88,8173 ± 3,6515 % |
| | **Test** | **88,2820 ± 4,4433 %** | **91,3883 ± 2,1025 %** | **14,8243 ± 9,6527 %** | **86,6958 ± 7,2180 %** | **88,7913 ± 3,6597 %** |

### One-Class SVM

| Parámetro | Conjunto | Accuracy | Recall | FPR | Precision | F1 |
|---|---|---:|---:|---:|---:|---:|
| q 0,9000 ± 0,0000 (fijo) | Validación | 78,0974 ± 0,0591 % | 69,5716 ± 0,2390 % | 13,3767 ± 0,1984 % | 83,8741 ± 0,1621 % | 76,0558 ± 0,0989 % |
| | **Test** | **78,0002 ± 0,0524 %** | **69,7872 ± 0,1731 %** | **13,7869 ± 0,1735 %** | **83,5038 ± 0,1468 %** | **76,0316 ± 0,0714 %** |

## Resultados por ejecución (test)

### SAD — NF-CSE-CIC-IDS2018

| Semilla | Umbral | F1 validación | FPR validación | Accuracy test | Recall test | FPR test | Precision test | F1 test |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 3,0σ | 90,9069 % | 6,8037 % | 91,2154 % | 89,1806 % | 6,7498 % | 92,9639 % | 91,0329 % |
| 1 | 3,0σ | 90,9548 % | 6,8944 % | 91,1893 % | 89,1753 % | 6,7967 % | 92,9180 % | 91,0082 % |
| 2 | 3,0σ | 90,9282 % | 6,7981 % | 91,1432 % | 89,1421 % | 6,8558 % | 92,8584 % | 90,9623 % |
| 3 | 3,0σ | 90,9589 % | 6,8736 % | 91,1784 % | 89,2047 % | 6,8479 % | 92,8707 % | 91,0008 % |
| 4 | 3,0σ | 90,9007 % | 6,8661 % | 91,1574 % | 89,1262 % | 6,8114 % | 92,9002 % | 90,9741 % |
| 5 | 3,0σ | 90,9254 % | 6,8238 % | 91,1846 % | 89,1419 % | 6,7727 % | 92,9388 % | 91,0007 % |
| 6 | 3,0σ | 90,9319 % | 6,8593 % | 91,2192 % | 89,2112 % | 6,7727 % | 92,9439 % | 91,0393 % |
| 7 | 3,0σ | 90,9014 % | 6,8357 % | 91,2176 % | 89,1492 % | 6,7139 % | 92,9964 % | 91,0321 % |
| 8 | 3,0σ | 90,9479 % | 6,8126 % | 91,1954 % | 89,1529 % | 6,7622 % | 92,9499 % | 91,0118 % |
| 9 | 3,0σ | 90,9148 % | 6,8063 % | 91,2132 % | 89,2404 % | 6,8140 % | 92,9061 % | 91,0364 % |
| 10 | 3,0σ | 90,9468 % | 6,8754 % | 91,2206 % | 89,2485 % | 6,8073 % | 92,9132 % | 91,0440 % |
| 11 | 3,0σ | 90,9085 % | 6,8681 % | 91,2127 % | 89,2663 % | 6,8408 % | 92,8821 % | 91,0383 % |

### LODA — NF-CSE-CIC-IDS2018

| Semilla | q | F1 validación | FPR validación | Accuracy test | Recall test | FPR test | Precision test | F1 test |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0,95 | 93,6626 % | 4,8975 % | 93,7345 % | 92,3548 % | 4,8858 % | 94,9756 % | 93,6469 % |
| 1 | 0,85 | 87,0822 % | 14,9470 % | 86,8589 % | 88,6554 % | 14,9376 % | 85,5805 % | 87,0908 % |
| 2 | 0,90 | 88,9698 % | 10,5174 % | 89,0204 % | 88,5638 % | 10,5229 % | 89,3801 % | 88,9701 % |
| 3 | 0,95 | 91,5429 % | 4,8958 % | 91,8207 % | 88,5558 % | 4,9144 % | 94,7422 % | 91,5446 % |
| 4 | 0,70 | 83,3763 % | 30,0362 % | 81,3952 % | 92,9333 % | 30,1430 % | 75,5087 % | 83,3197 % |
| 5 | 0,90 | 92,5192 % | 7,3740 % | 92,5245 % | 92,3990 % | 7,3499 % | 92,6316 % | 92,5151 % |
| 6 | 0,70 | 83,7858 % | 29,7199 % | 81,8350 % | 93,4710 % | 29,8010 % | 75,8250 % | 83,7284 % |
| 7 | 0,90 | 91,4717 % | 9,6516 % | 91,3486 % | 92,3846 % | 9,6874 % | 90,5093 % | 91,4373 % |
| 8 | 0,70 | 83,8035 % | 30,0856 % | 81,8292 % | 93,7816 % | 30,1232 % | 75,6884 % | 83,7692 % |
| 9 | 0,90 | 91,0637 % | 10,6263 % | 90,8914 % | 92,4372 % | 10,6544 % | 89,6651 % | 91,0301 % |
| 10 | 0,90 | 91,0639 % | 10,6626 % | 90,8357 % | 92,4702 % | 10,7988 % | 89,5430 % | 90,9831 % |
| 11 | 0,85 | 87,4659 % | 14,0437 % | 87,2897 % | 88,6527 % | 14,0734 % | 86,3001 % | 87,4606 % |

### One-Class SVM — NF-CSE-CIC-IDS2018

| Semilla | q | F1 validación | FPR validación | Accuracy test | Recall test | FPR test | Precision test | F1 test |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0,90 | 76,1653 % | 13,6430 % | 78,0650 % | 70,0140 % | 13,8840 % | 83,4513 % | 76,1444 % |
| 1 | 0,90 | 76,0376 % | 13,4715 % | 77,9832 % | 69,6843 % | 13,7180 % | 83,5520 % | 75,9907 % |
| 2 | 0,90 | 76,0855 % | 13,4719 % | 78,0242 % | 69,9920 % | 13,9437 % | 83,3877 % | 76,1049 % |
| 3 | 0,90 | 76,0465 % | 13,2523 % | 78,0088 % | 69,6500 % | 13,6324 % | 83,6311 % | 76,0029 % |
| 4 | 0,90 | 76,1737 % | 13,3655 % | 77,9665 % | 69,7620 % | 13,8291 % | 83,4563 % | 75,9972 % |
| 5 | 0,90 | 75,8912 % | 13,2303 % | 77,9590 % | 69,5379 % | 13,6199 % | 83,6217 % | 75,9323 % |
| 6 | 0,90 | 76,0325 % | 13,5068 % | 77,9921 % | 69,9067 % | 13,9225 % | 83,3919 % | 76,0561 % |
| 7 | 0,90 | 76,0430 % | 13,0315 % | 78,0814 % | 69,8568 % | 13,6940 % | 83,6099 % | 76,1171 % |
| 8 | 0,90 | 76,1036 % | 13,6428 % | 77,9948 % | 70,0368 % | 14,0471 % | 83,2940 % | 76,0922 % |
| 9 | 0,90 | 75,9216 % | 13,0806 % | 78,0624 % | 69,6133 % | 13,4884 % | 83,7688 % | 76,0378 % |
| 10 | 0,90 | 76,2085 % | 13,4829 % | 77,8972 % | 69,7864 % | 13,9919 % | 83,2989 % | 75,9463 % |
| 11 | 0,90 | 75,9609 % | 13,3414 % | 77,9673 % | 69,6068 % | 13,6723 % | 83,5826 % | 75,9572 % |

## Tiempos (media ± desviación típica, n = 12)

### SAD

| Fase | Tiempo |
|---|---:|
| Búsqueda del umbral (24 candidatos) | 149,7410 ± 4,0684 s |
| Entrenamiento final (100.000 flujos) | 0,1961 ± 0,0030 s |
| Evaluación de test | 5,5309 ± 0,3748 s |
| Coste por flujo en test | 0,0054 ± 0,0004 ms |
| Ejecución completa | 205,56 ± 4,75 s |

### LODA

| Fase | Tiempo |
|---|---:|
| Entrenamiento (100.000 flujos) | 6,1062 ± 0,0766 s |
| Scoring de entrenamiento | 7,2321 ± 0,1089 s |
| Scoring de validación | 73,1631 ± 1,0776 s |
| Selección del umbral (8 cuantiles) | 0,3672 ± 0,0079 s |
| Scoring de test | 73,1895 ± 1,8231 s |
| Coste por flujo en test | 0,0718 ± 0,0018 ms |
| Ejecución completa | 211,04 ± 2,40 s |

### One-Class SVM

| Fase | Tiempo |
|---|---:|
| Entrenamiento (100.000 flujos) | 0,5337 ± 0,0086 s |
| Evaluación de validación | 4,7882 ± 0,0558 s |
| Evaluación de test | 4,8836 ± 0,0535 s |
| Coste por flujo en test | 0,0048 ± 0,0001 ms |
| Ejecución completa | 61,42 ± 0,58 s |

## Comparación con las ejecuciones individuales previas

| Modelo | Ejecución individual | Media multisemilla (n = 12) | Diferencia de F1 |
|---|---|---|---:|
| SAD | F1 90,9864 %, FPR 6,7994 % (umbral 3,0σ) | F1 91,0151 ± 0,0269 %, FPR 6,7954 ± 0,0427 % | +0,0287 |
| LODA | F1 83,9354 %, FPR 30,1701 % (q = 0,85) | F1 88,7913 ± 3,6597 %, FPR 14,8243 ± 9,6527 % | +4,8559 |
| One-Class SVM | F1 76,0385 %, FPR 13,5096 % (Standard, ν=0.5, q=0.9, lr=0.3) | F1 76,0316 ± 0,0714 %, FPR 13,7869 ± 0,1735 % | −0,0069 |

SAD y One-Class SVM son reproducibles: la ejecución individual queda dentro de
la dispersión de las 12 repeticiones. LODA no: la ejecución individual publicada
(F1 83,94 %) está en el extremo inferior del rango observado
(F1 83,32 %–93,65 %), por lo que subestimaba el rendimiento medio de este
protocolo.

## Comparación con UNSW

| Dataset | Modelo | Accuracy | Recall | FPR | F1 |
|---|---|---:|---:|---:|---:|
| NF-UNSW-NB15 | SAD | 97,6051 ± 0,3651 % | 100,0000 % | 4,7899 ± 0,7302 % | 97,6622 ± 0,3475 % |
| NF-UNSW-NB15 | LODA | 95,0337 ± 0,4223 % | 99,6962 ± 1,0525 % | 9,6288 ± 1,3981 % | 95,2561 ± 0,3819 % |
| NF-UNSW-NB15 | One-Class SVM | 97,7703 ± 0,1057 % | 99,5007 ± 0,2249 % | 3,9601 ± 0,0140 % | 97,8081 ± 0,1064 % |
| NF-UNSW-NB15-v2 | SAD | 99,8894 ± 0,0047 % | 100,0000 % | 0,2213 ± 0,0094 % | 99,8895 ± 0,0047 % |
| NF-UNSW-NB15-v2 | LODA | 98,8770 ± 0,3137 % | 99,9998 ± 0,0006 % | 2,2459 ± 0,6279 % | 98,8903 ± 0,3091 % |
| NF-UNSW-NB15-v2 | One-Class SVM | 99,7514 ± 0,4200 % | 99,5210 ± 0,8463 % | 0,0182 ± 0,0082 % | 99,7491 ± 0,4254 % |
| **NF-CSE-CIC-IDS2018** | **SAD** | **91,1956 ± 0,0258 %** | **89,1866 ± 0,0470 %** | **6,7954 ± 0,0427 %** | **91,0151 ± 0,0269 %** |
| NF-CSE-CIC-IDS2018 | LODA | 88,2820 ± 4,4433 % | 91,3883 ± 2,1025 % | 14,8243 ± 9,6527 % | 88,7913 ± 3,6597 % |
| NF-CSE-CIC-IDS2018 | One-Class SVM | 78,0002 ± 0,0524 % | 69,7872 ± 0,1731 % | 13,7869 ± 0,1735 % | 76,0316 ± 0,0714 % |

Los valores de UNSW proceden de
`resultados_multisemilla_sad_loda_unsw.md` y
`resultados_split_entrenamiento_validacion_test.md`.

## Observaciones

1. **SAD es el mejor modelo en CIC y el más estable.** F1 91,0151 % ± 0,0269 %,
   FPR 6,7954 % ± 0,0427 % y recall 89,1866 % ± 0,0470 %. El umbral elegido es
   3,0σ en las 12 ejecuciones. No cumple los objetivos del paper
   (accuracy > 98 %, FPR < 3,1 %, recall 100 %), pero es el que más se acerca.

2. **LODA es muy variable en CIC.** F1 88,7913 % ± 3,6597 %, FPR
   14,8243 % ± 9,6527 % y q 0,8500 ± 0,0953. El rango de F1 va de 83,32 % a
   93,65 % y el de FPR de 4,89 % a 30,14 %. La causa principal es la selección
   del cuantil en validación: según el orden interno, el mejor q cambia entre
   0,70 y 0,95, y con él cambia radicalmente el punto de operación.

3. **La ejecución individual de LODA era pesimista.** El valor publicado
   (F1 83,94 %, FPR 30,17 %) corresponde a un q = 0,85 con un orden concreto.
   Con 12 semillas la media sube a F1 88,79 % y el FPR baja a 14,82 %, aunque
   con una dispersión muy alta. Reportar una sola ejecución no es representativo
   para LODA en CIC.

4. **One-Class SVM es el peor modelo en CIC.** F1 76,0316 % ± 0,0714 % y recall
   69,7872 % ± 0,1731 %. Es muy estable, pero no detecta ~30 % de las anomalías
   y marca un 13,79 % de los benignos. Los hiperparámetros del artículo para
   UNSW no se transfieren a CIC.

5. **Ningún modelo cumple los objetivos del paper en CIC.** El mejor F1 es
   91,02 % (SAD), lejos del > 98 % de accuracy y del FPR < 3,1 % exigidos.

6. **El orden de aprendizaje importa mucho más en LODA que en SAD y SVM.** En
   SAD la desviación del F1 es 0,0269 puntos y en el SVM 0,0714; en LODA es
   3,6597. Esto refuerza la necesidad de reportar media ± desviación en CIC y no
   una sola ejecución.

7. **La desviación no representa la variabilidad del split.** Al mantener fija
   la partición (CSV barajado con semilla 42), las 12 ejecuciones solo cambian
   el orden interno. Como CIC es muy sensible al particionado, esta es la
   principal limitación del experimento: para obtener barras de error
   representativas del split habría que barajar el CSV con una semilla distinta
   en cada ejecución y repetir las 36 ejecuciones.

8. **Validación y test coinciden.** En SAD y SVM las métricas de validación y
   test son prácticamente iguales, lo que confirma que el split aleatorio es
   estable. En LODA también coinciden las medias, pero la dispersión es alta en
   ambos conjuntos.

9. **Validación y test están balanceados al 50 %.** El FPR y el recall deben
   interpretarse con esa proporción; en tráfico real, mucho más desbalanceado,
   la precisión sería menor.

## Reproducibilidad

Crear el CSV barajado (si no existe):

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

Ejecutar las 12 semillas de cada modelo (pueden lanzarse en paralelo):

```bash
cd "/home/alberto/Online machine learning/codigo-paper-oml-computer-security"

.venv/bin/python multiexecution_cic.py --models SAD --num-executions 12 --seed-start 0
.venv/bin/python multiexecution_cic.py --models LODA --num-executions 12 --seed-start 0
.venv/bin/python multiexecution_cic.py --models OCSVM --num-executions 12 --seed-start 0

.venv/bin/python merge_cic_summaries.py
```

Opciones disponibles en `multiexecution_cic.py`:

| Opción | Por defecto | Descripción |
|---|---|---|
| `--num-executions` | `12` | Número de semillas por modelo |
| `--seed-start` | `0` | Primera semilla; se usan enteros consecutivos |
| `--models` | `SAD LODA OCSVM` | Modelos a ejecutar |
| `--datasets` | `NF-CSE-CIC-IDS2018` | Dataset a ejecutar |
| `--output-dir` | `resultados` | Directorio de salida |
| `--summary-name` | derivado de los modelos | Nombre del JSON agregado |
| `--skip-existing` | desactivado | Reutiliza los JSON por ejecución ya existentes |

Salidas:

- `resultados/multisemilla_cic_ids2018.json`: resumen agregado con medias,
  desviaciones, tiempos y resultados por ejecución.
- `resultados/multisemilla_cic_ids2018_SAD.json`,
  `resultados/multisemilla_cic_ids2018_LODA.json` y
  `resultados/multisemilla_cic_ids2018_OCSVM.json`: resúmenes por modelo.
- `resultados/multisemilla_cic_ids2018/raw/`: un JSON por modelo y semilla
  (36 ficheros), con métricas de validación y test y tiempos.

Cada ejecución se lanza como un proceso independiente y fija la semilla de forma
explícita, por lo que las tablas se reproducen con las mismas versiones de las
librerías.

## Referencias

- Resultados de una ejecución de SAD en CIC: `resultados_sad_cic_ids2018.md`.
- Resultados de una ejecución de LODA en CIC: `resultados_loda_cic_ids2018.md`.
- Resultados de una ejecución de One-Class SVM en CIC:
  `resultados_ocsvm_cic_ids2018.md`.
- Resultados multisemilla en UNSW:
  `resultados_multisemilla_sad_loda_unsw.md`.
- Pipeline One-Class SVM en UNSW:
  `resultados_split_entrenamiento_validacion_test.md`.
