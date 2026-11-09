# Repeticiones multisemilla de SAD y LODA sobre NF-UNSW-NB15 y NF-UNSW-NB15-v2

Evaluación de `river.anomaly.StandardAbsoluteDeviation` (SAD) y
`river.anomaly.LODA` con 12 ejecuciones por modelo y dataset, siguiendo el mismo
esquema que `multiexecution.py`: la pertenencia de cada flujo a los conjuntos de
escalador / entrenamiento / validación / test es fija, y cada semilla cambia
únicamente el orden interno de esos conjuntos. En LODA la misma semilla controla
además las proyecciones aleatorias del modelo.

El objetivo es completar los resultados de una sola ejecución de
`resultados_sad_unsw.md`, `resultados_loda_unsw.md` y
`resultados_loda_unsw_v2.md` con media ± desviación típica, igual que hace el
pipeline principal con One-Class SVM.

## Metadatos

| Campo | Valor |
|---|---|
| Fecha de ejecución | 2026-11-03 |
| Rama | `feature/more-models` |
| Commit | `dfde77c` + cambios locales (`sad_detector.py`, `multiexecution_sad_loda.py`, este documento) |
| Script | `multiexecution_sad_loda.py` |
| Ejecuciones | 48 (12 semillas × 2 modelos × 2 datasets) |
| Semillas | `0`–`11` |
| Duración total | 2.693,02 s (44 min 53 s) |
| Python | 3.12.3 |
| river | 0.26.1 |
| pandas | 3.1.0 |
| numpy | 2.5.4 |
| scikit-learn | 1.9.1 |
| matplotlib | 3.11.2 |
| PyYAML | 6.0.3 |

## Protocolo

Cada ejecución es un proceso independiente que llama al script del modelo con
`--seed <s>` y `--output-json`. El protocolo interno es el ya documentado:

- **SAD**: un detector por característica, agregación `max`, `MaxAbsScaler`,
  umbral elegido en validación sobre una rejilla de 0,0–10,0σ con paso 0,1
  (104 candidatos), modelo reevaluado online y test evaluado una sola vez.
- **LODA**: entrenamiento directo con los 100.000 flujos benignos,
  `StandardScaler`, `n_bins=10`, `n_random_cuts=100`, umbral elegido como
  cuantil de los scores de entrenamiento sobre la rejilla
  `0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99`, detector congelado en validación
  y test.

Parámetros comunes:

| Parámetro | Valor |
|---|---|
| `FLOWS_TRAIN_SCALER` | 1.000 |
| `FLOWS_TRAIN_OML` | 100.000 |
| `VALIDATION_RATIO` | 0.5 |
| `SELECTION_METRIC` | `f1` |
| Particiones | Las mismas de `hyperparameter_optimization.py` (ver tabla siguiente) |

| Dataset | Flujos totales | Benignos | Anomalías | Validación (benignos/anomalías) | Test (benignos/anomalías) |
|---|---:|---:|---:|---:|---:|
| NF-UNSW-NB15 | 1.623.118 | 1.550.712 | 72.406 | 36.203 / 36.203 | 36.203 / 36.203 |
| NF-UNSW-NB15-v2 | 2.390.275 | 2.295.222 | 95.053 | 47.526 / 47.526 | 47.527 / 47.527 |

> **Alcance de la desviación típica.** Como en `multiexecution.py`, la partición
> no se remuestrea entre semillas: cada flujo pertenece siempre al mismo
> conjunto. La desviación mide el efecto del orden de aprendizaje online (y, en
> LODA, también el de las proyecciones aleatorias), no la variabilidad del split.
> Para medir esta última habría que barajar el CSV con una semilla distinta en
> cada ejecución.

## Resultados agregados (media ± desviación típica, n = 12)

### SAD

| Dataset | Parámetro | Conjunto | Accuracy | Recall | FPR | Precision | F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| NF-UNSW-NB15 | umbral 4,8250σ ± 0,1288σ | Validación | 98,0871 ± 0,4059 % | 100,0000 ± 0,0000 % | 3,8259 ± 0,8118 % | 96,3205 ± 0,7498 % | 98,1244 ± 0,3899 % |
| NF-UNSW-NB15 | | **Test** | **97,6051 ± 0,3651 %** | **100,0000 ± 0,0000 %** | **4,7899 ± 0,7302 %** | **95,4333 ± 0,6622 %** | **97,6622 ± 0,3475 %** |
| NF-UNSW-NB15-v2 | umbral 6,2500σ ± 0,7416σ | Validación | 99,8176 ± 0,0044 % | 100,0000 ± 0,0000 % | 0,3647 ± 0,0087 % | 99,6366 ± 0,0087 % | 99,8180 ± 0,0043 % |
| NF-UNSW-NB15-v2 | | **Test** | **99,8894 ± 0,0047 %** | **100,0000 ± 0,0000 %** | **0,2213 ± 0,0094 %** | **99,7792 ± 0,0094 %** | **99,8895 ± 0,0047 %** |

### LODA

| Dataset | Parámetro | Conjunto | Accuracy | Recall | FPR | Precision | F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| NF-UNSW-NB15 | q 0,9042 ± 0,0144 | Validación | 96,0733 ± 0,4651 % | 99,6713 ± 1,1387 % | 7,5247 ± 1,4501 % | 93,0050 ± 1,2380 % | 96,2109 ± 0,4337 % |
| NF-UNSW-NB15 | | **Test** | **95,0337 ± 0,4223 %** | **99,6962 ± 1,0525 %** | **9,6288 ± 1,3981 %** | **91,2137 ± 1,1336 %** | **95,2561 ± 0,3819 %** |
| NF-UNSW-NB15-v2 | q 0,9533 ± 0,0115 | Validación | 98,6617 ± 0,3714 % | 99,9995 ± 0,0018 % | 2,6761 ± 0,7444 % | 97,3984 ± 0,7178 % | 98,6806 ± 0,3646 % |
| NF-UNSW-NB15-v2 | | **Test** | **98,8770 ± 0,3137 %** | **99,9998 ± 0,0006 %** | **2,2459 ± 0,6279 %** | **97,8068 ± 0,6098 %** | **98,8903 ± 0,3091 %** |

## Resultados por ejecución (test)

### SAD — NF-UNSW-NB15

| Semilla | Umbral | F1 validación | FPR validación | Accuracy test | Recall test | FPR test | Precision test | F1 test |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 4,5σ | 98,4218 % | 3,2069 % | 97,8800 % | 100,0000 % | 4,2400 % | 95,9325 % | 97,9240 % |
| 1 | 4,9σ | 97,5559 % | 5,0106 % | 97,0707 % | 100,0000 % | 5,8586 % | 94,4656 % | 97,1541 % |
| 2 | 4,9σ | 98,4218 % | 3,2069 % | 97,8800 % | 100,0000 % | 4,2400 % | 95,9325 % | 97,9240 % |
| 3 | 4,9σ | 97,5546 % | 5,0134 % | 97,0610 % | 100,0000 % | 5,8780 % | 94,4484 % | 97,1449 % |
| 4 | 4,9σ | 98,4085 % | 3,2345 % | 97,8676 % | 100,0000 % | 4,2648 % | 95,9096 % | 97,9121 % |
| 5 | 4,9σ | 97,5993 % | 4,9195 % | 97,1121 % | 100,0000 % | 5,7758 % | 94,5396 % | 97,1932 % |
| 6 | 4,8σ | 98,4218 % | 3,2069 % | 97,8800 % | 100,0000 % | 4,2400 % | 95,9325 % | 97,9240 % |
| 7 | 4,9σ | 98,4433 % | 3,1627 % | 97,8980 % | 100,0000 % | 4,2041 % | 95,9655 % | 97,9412 % |
| 8 | 4,7σ | 98,4218 % | 3,2069 % | 97,8814 % | 100,0000 % | 4,2372 % | 95,9350 % | 97,9253 % |
| 9 | 4,7σ | 98,4218 % | 3,2069 % | 97,8800 % | 100,0000 % | 4,2400 % | 95,9325 % | 97,9240 % |
| 10 | 4,9σ | 97,7825 % | 4,5355 % | 97,2778 % | 100,0000 % | 5,4443 % | 94,8368 % | 97,3500 % |
| 11 | 4,9σ | 98,0394 % | 3,9997 % | 97,5720 % | 100,0000 % | 4,8560 % | 95,3689 % | 97,6296 % |

### SAD — NF-UNSW-NB15-v2

| Semilla | Umbral | F1 validación | FPR validación | Accuracy test | Recall test | FPR test | Precision test | F1 test |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 5,3σ | 99,8162 % | 0,3682 % | 99,8874 % | 100,0000 % | 0,2251 % | 99,7754 % | 99,8876 % |
| 1 | 5,7σ | 99,8162 % | 0,3682 % | 99,8874 % | 100,0000 % | 0,2251 % | 99,7754 % | 99,8876 % |
| 2 | 7,4σ | 99,8309 % | 0,3388 % | 99,9022 % | 100,0000 % | 0,1957 % | 99,8047 % | 99,9023 % |
| 3 | 5,7σ | 99,8162 % | 0,3682 % | 99,8874 % | 100,0000 % | 0,2251 % | 99,7754 % | 99,8876 % |
| 4 | 6,5σ | 99,8162 % | 0,3682 % | 99,8874 % | 100,0000 % | 0,2251 % | 99,7754 % | 99,8876 % |
| 5 | 5,7σ | 99,8162 % | 0,3682 % | 99,8874 % | 100,0000 % | 0,2251 % | 99,7754 % | 99,8876 % |
| 6 | 6,3σ | 99,8162 % | 0,3682 % | 99,8874 % | 100,0000 % | 0,2251 % | 99,7754 % | 99,8876 % |
| 7 | 6,4σ | 99,8162 % | 0,3682 % | 99,8874 % | 100,0000 % | 0,2251 % | 99,7754 % | 99,8876 % |
| 8 | 7,2σ | 99,8215 % | 0,3577 % | 99,8958 % | 100,0000 % | 0,2083 % | 99,7921 % | 99,8960 % |
| 9 | 7,4σ | 99,8173 % | 0,3661 % | 99,8874 % | 100,0000 % | 0,2251 % | 99,7754 % | 99,8876 % |
| 10 | 5,7σ | 99,8162 % | 0,3682 % | 99,8874 % | 100,0000 % | 0,2251 % | 99,7754 % | 99,8876 % |
| 11 | 5,7σ | 99,8162 % | 0,3682 % | 99,8874 % | 100,0000 % | 0,2251 % | 99,7754 % | 99,8876 % |

### LODA — NF-UNSW-NB15

| Semilla | q | F1 validación | FPR validación | Accuracy test | Recall test | FPR test | Precision test | F1 test |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0,90 | 95,3714 % | 9,7064 % | 94,1055 % | 100,0000 % | 11,7891 % | 89,4542 % | 94,4336 % |
| 1 | 0,90 | 95,9668 % | 8,4054 % | 94,8651 % | 100,0000 % | 10,2699 % | 90,6866 % | 95,1159 % |
| 2 | 0,90 | 96,2104 % | 7,8778 % | 94,9604 % | 100,0000 % | 10,0793 % | 90,8436 % | 95,2022 % |
| 3 | 0,90 | 96,9550 % | 6,2812 % | 95,6937 % | 100,0000 % | 8,6125 % | 92,0704 % | 95,8715 % |
| 4 | 0,90 | 95,7131 % | 8,9578 % | 94,5460 % | 100,0000 % | 10,9079 % | 90,1649 % | 94,8281 % |
| 5 | 0,90 | 96,1235 % | 8,0656 % | 95,1275 % | 100,0000 % | 9,7450 % | 91,1203 % | 95,3539 % |
| 6 | 0,90 | 96,3268 % | 7,6264 % | 95,1192 % | 100,0000 % | 9,7616 % | 91,1065 % | 95,3463 % |
| 7 | 0,90 | 96,2628 % | 7,7645 % | 94,9728 % | 100,0000 % | 10,0544 % | 90,8641 % | 95,2134 % |
| 8 | 0,90 | 96,7258 % | 6,7702 % | 95,4189 % | 100,0000 % | 9,1622 % | 91,6068 % | 95,6196 % |
| 9 | 0,90 | 96,2040 % | 7,8916 % | 94,9686 % | 100,0000 % | 10,0627 % | 90,8573 % | 95,2097 % |
| 10 | 0,90 | 96,6380 % | 6,9580 % | 95,4921 % | 100,0000 % | 9,0158 % | 91,7298 % | 95,6865 % |
| 11 | 0,95 | 96,0330 % | 3,9914 % | 95,1344 % | 96,3539 % | 6,0851 % | 94,0598 % | 95,1930 % |

### LODA — NF-UNSW-NB15-v2

| Semilla | q | F1 validación | FPR validación | Accuracy test | Recall test | FPR test | Precision test | F1 test |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0,95 | 98,6682 % | 2,6996 % | 98,7986 % | 100,0000 % | 2,4028 % | 97,6535 % | 98,8128 % |
| 1 | 0,95 | 98,3670 % | 3,3203 % | 98,6902 % | 100,0000 % | 2,6196 % | 97,4473 % | 98,7072 % |
| 2 | 0,95 | 98,6938 % | 2,6470 % | 98,8775 % | 100,0000 % | 2,2450 % | 97,8043 % | 98,8899 % |
| 3 | 0,99 | 99,7565 % | 0,4818 % | 99,8296 % | 99,9979 % | 0,3388 % | 99,6624 % | 99,8299 % |
| 4 | 0,95 | 98,5516 % | 2,9394 % | 98,8112 % | 100,0000 % | 2,3776 % | 97,6776 % | 98,8252 % |
| 5 | 0,95 | 98,5270 % | 2,9899 % | 98,7649 % | 100,0000 % | 2,4702 % | 97,5894 % | 98,7800 % |
| 6 | 0,95 | 98,4719 % | 3,1036 % | 98,7544 % | 100,0000 % | 2,4912 % | 97,5693 % | 98,7697 % |
| 7 | 0,95 | 98,4271 % | 3,1961 % | 98,5850 % | 100,0000 % | 2,8300 % | 97,2479 % | 98,6048 % |
| 8 | 0,95 | 98,5526 % | 2,9373 % | 98,8112 % | 100,0000 % | 2,3776 % | 97,6776 % | 98,8252 % |
| 9 | 0,95 | 98,8735 % | 2,2788 % | 98,9511 % | 100,0000 % | 2,0978 % | 97,9453 % | 98,9620 % |
| 10 | 0,95 | 98,6180 % | 2,8027 % | 98,7996 % | 100,0000 % | 2,4007 % | 97,6555 % | 98,8139 % |
| 11 | 0,95 | 98,6600 % | 2,7164 % | 98,8501 % | 100,0000 % | 2,2997 % | 97,7520 % | 98,8632 % |

## Tiempos (media ± desviación típica, n = 12)

### SAD

| Dataset | Búsqueda del umbral | Entrenamiento final | Test | Coste por flujo en test | Ejecución completa |
|---|---:|---:|---:|---:|---:|
| NF-UNSW-NB15 | 61,2545 ± 0,5415 s | 0,1911 ± 0,0038 s | 0,3764 ± 0,0160 s | 0,0052 ± 0,0002 ms | 70,04 ± 0,57 s |
| NF-UNSW-NB15-v2 | 72,6410 ± 0,6755 s | 0,1892 ± 0,0034 s | 0,5005 ± 0,0157 s | 0,0053 ± 0,0002 ms | 84,38 ± 0,71 s |

### LODA

| Dataset | Entrenamiento | Scoring de entrenamiento | Scoring de validación | Selección del umbral | Scoring de test | Coste por flujo en test | Ejecución completa |
|---|---:|---:|---:|---:|---:|---:|---:|
| NF-UNSW-NB15 | 6,0304 ± 0,0460 s | 7,1017 ± 0,1045 s | 5,1351 ± 0,0685 s | 0,0300 ± 0,0011 s | 5,1056 ± 0,0706 s | 0,0705 ± 0,0010 ms | 31,86 ± 0,19 s |
| NF-UNSW-NB15-v2 | 6,0252 ± 0,2162 s | 7,1066 ± 0,1021 s | 6,7462 ± 0,0858 s | 0,0371 ± 0,0004 s | 6,7267 ± 0,1132 s | 0,0708 ± 0,0012 ms | 38,14 ± 0,20 s |

## Comparación con el pipeline One-Class SVM

Los valores de One-Class SVM proceden de
`resultados_split_entrenamiento_validacion_test.md` (12 ejecuciones, media ±
desviación típica). Ahora ambos experimentos siguen el mismo esquema de
repeticiones: partición fija y solo cambia el orden interno. La comparación es
más directa que antes, aunque las versiones de las librerías no son idénticas
(river 0.21.1 frente a 0.26.1).

| Dataset | Modelo | Accuracy | Recall | FPR | F1 |
|---|---|---:|---:|---:|---:|
| NF-UNSW-NB15 | One-Class SVM | 97,7703 ± 0,1057 % | 99,5007 ± 0,2249 % | **3,9601 ± 0,0140 %** | 97,8081 ± 0,1064 % |
| NF-UNSW-NB15 | SAD | 97,6051 ± 0,3651 % | **100,0000 ± 0,0000 %** | 4,7899 ± 0,7302 % | 97,6622 ± 0,3475 % |
| NF-UNSW-NB15 | LODA | 95,0337 ± 0,4223 % | 99,6962 ± 1,0525 % | 9,6288 ± 1,3981 % | 95,2561 ± 0,3819 % |
| NF-UNSW-NB15-v2 | One-Class SVM | 99,7514 ± 0,4200 % | 99,5210 ± 0,8463 % | **0,0182 ± 0,0082 %** | 99,7491 ± 0,4254 % |
| NF-UNSW-NB15-v2 | **SAD** | **99,8894 ± 0,0047 %** | **100,0000 ± 0,0000 %** | 0,2213 ± 0,0094 % | **99,8895 ± 0,0047 %** |
| NF-UNSW-NB15-v2 | LODA | 98,8770 ± 0,3137 % | 99,9998 ± 0,0006 % | 2,2459 ± 0,6279 % | 98,8903 ± 0,3091 % |

## Comparación con las ejecuciones individuales previas

| Dataset | Modelo | Ejecución individual | Media multisemilla | Diferencia de F1 |
|---|---|---|---:|---:|
| NF-UNSW-NB15 | SAD | F1 97,2898 %, FPR 5,5714 % (4,9σ) | F1 97,6622 ± 0,3475 %, FPR 4,7899 ± 0,7302 % | +0,3724 |
| NF-UNSW-NB15-v2 | SAD | F1 99,8876 %, FPR 0,2251 % (5,7σ) | F1 99,8895 ± 0,0047 %, FPR 0,2213 ± 0,0094 % | +0,0019 |
| NF-UNSW-NB15 | LODA | F1 95,1634 %, FPR 10,1649 % | F1 95,2561 ± 0,3819 %, FPR 9,6288 ± 1,3981 % | +0,0927 |
| NF-UNSW-NB15-v2 | LODA | F1 98,7000 %, FPR 2,6343 % | F1 98,8903 ± 0,3091 %, FPR 2,2459 ± 0,6279 % | +0,1903 |

Las diferencias son pequeñas: las ejecuciones individuales publicadas son
representativas del rendimiento medio, pero no capturan la dispersión entre
órdenes de aprendizaje.

## Observaciones

1. **SAD es muy estable en UNSW v2.** La desviación típica del F1 es 0,0047
   puntos y la del FPR 0,0094 puntos. Con 12 semillas se mantiene recall 100 %,
   accuracy 99,8894 % y FPR 0,2213 %, cumpliendo los objetivos del paper.
2. **SAD no cumple el objetivo de FPR en UNSW v1.** La media es 4,7899 % ±
   0,7302 %, por encima del 3,1 %. El umbral elegido varía entre 4,5σ y 4,9σ y
   el FPR de test entre 4,20 % y 5,88 %, así que el resultado es sensible al
   orden de validación.
3. **LODA cumple los objetivos en UNSW v2 de media, pero con más dispersión.**
   F1 98,8903 % ± 0,3091 %, FPR 2,2459 % ± 0,6279 % y recall prácticamente
   100 %. Once de las doce semillas eligen q = 0,95; la semilla 3 elige q = 0,99
   y obtiene el mejor resultado (F1 99,8299 %, FPR 0,3388 %).
4. **LODA sigue sin cumplir en UNSW v1.** F1 95,2561 % ± 0,3819 %, FPR
   9,6288 % ± 1,3981 %. Además, la semilla 11 elige q = 0,95 y el recall cae a
   96,3539 %, lo que baja la media de recall a 99,6962 % ± 1,0525 %. No hay un
   umbral que dé a la vez recall alto y FPR bajo.
5. **El orden de aprendizaje importa más que el modelo en algunos casos.** En
   SAD v1 el mismo umbral de 4,9σ produce FPR de test entre 4,20 % y 5,88 %
   según la semilla. En LODA v1 el FPR de test va de 6,09 % a 11,79 %. Por eso
   reportar una sola ejecución puede sobreestimar o subestimar el rendimiento.
6. **Comparación con One-Class SVM.** En v1, SAD y One-Class SVM quedan casi
   empatados en F1 (97,66 % frente a 97,81 %), pero el SVM tiene menor FPR
   (3,96 % frente a 4,79 %) y menor recall (99,50 % frente a 100 %). En v2, SAD
   supera al SVM en F1 medio (99,89 % frente a 99,75 %) y recall (100 % frente a
   99,52 %), pero el SVM mantiene un FPR casi nulo (0,02 % frente a 0,22 %).
   LODA queda por debajo de ambos en los dos datasets.
7. **La desviación no representa la variabilidad del split.** Al mantener fija
   la partición, las 12 ejecuciones solo cambian el orden interno. Para obtener
   barras de error representativas del particionado habría que barajar el CSV
   con una semilla distinta en cada ejecución y repetir el experimento.
8. **Validación y test están balanceados al 50 %.** El FPR y el recall deben
   interpretarse con esa proporción; en tráfico real, mucho más desbalanceado,
   la precisión sería menor.

## Reproducibilidad

Desde la raíz del repositorio:

```bash
# Las 48 ejecuciones (12 semillas × SAD/LODA × UNSW v1/v2)
.venv/bin/python multiexecution_sad_loda.py
```

Opciones disponibles:

| Opción | Por defecto | Descripción |
|---|---|---|
| `--num-executions` | `12` | Número de semillas por modelo y dataset |
| `--seed-start` | `0` | Primera semilla; se usan enteros consecutivos |
| `--models` | `SAD LODA` | Modelos a ejecutar |
| `--datasets` | `NF-UNSW-NB15 NF-UNSW-NB15-v2` | Datasets a ejecutar |
| `--output-dir` | `resultados` | Directorio de salida |
| `--skip-existing` | desactivado | Reutiliza los JSON por ejecución ya existentes |

Ejemplos:

```bash
# Solo SAD sobre UNSW v2 con 5 semillas
.venv/bin/python multiexecution_sad_loda.py \
  --models SAD --datasets NF-UNSW-NB15-v2 --num-executions 5

# Reanudar sin repetir las ejecuciones ya guardadas
.venv/bin/python multiexecution_sad_loda.py --skip-existing
```

Salidas:

- `resultados/multisemilla_sad_loda_unsw.json`: resumen agregado con medias,
  desviaciones, tiempos y resultados por ejecución.
- `resultados/multisemilla_sad_loda/raw/`: un JSON por modelo, dataset y semilla
  (48 ficheros), con métricas de validación y test y tiempos.

El script fija las semillas de forma explícita y cada ejecución se lanza como un
proceso independiente, por lo que las tablas se reproducen con las mismas
versiones de las librerías.

## Referencias

- Resultados de una ejecución de SAD: `resultados_sad_unsw.md`.
- Resultados de una ejecución de LODA en v1: `resultados_loda_unsw.md`.
- Resultados de una ejecución de LODA en v2: `resultados_loda_unsw_v2.md`.
- Resultados del pipeline One-Class SVM: `resultados_split_entrenamiento_validacion_test.md`.
