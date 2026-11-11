# LODA sobre NF-CSE-CIC-IDS2018 con protocolo online

Doce ejecuciones (semillas 0-11) de LODA sobre el CSV barajado de
NF-CSE-CIC-IDS2018. El umbral se selecciona en validación y el test se
evalúa una sola vez mientras el modelo se actualiza online con los flujos
que no marca como anómalos.

## Configuración

| Parámetro | Valor |
|---|---|
| `scaling_type` | `Normalizer` |
| `n_bins` | 5 |
| `n_random_cuts` | 100 |
| Flujos de warm-up | 100000 |
| Criterio de selección | F1 de validación |
| Rejilla de cuantiles | 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99 |
| Semillas | 0-11 |

La configuración se eligió en un barrido previo de 45 combinaciones
(5 scalers × 3 valores de `n_bins` × 3 de `n_random_cuts`) con la semilla
42 (`resultados/barrido_loda_cic/`).

## Resultados agregados (media ± desviación típica)

| Split | Accuracy | Recall | FPR | Precision | F1 |
|---|---:|---:|---:|---:|---:|
| Validación | 90.4025 ± 4.2028 % | 91.9228 ± 2.0002 % | 11.1178 ± 9.5557 % | 89.8450 ± 6.5847 % | 90.6951 ± 3.2472 % |
| Test | 90.3442 ± 4.1900 % | 91.8797 ± 1.9985 % | 11.1913 ± 9.5363 % | 89.7759 ± 6.5839 % | 90.6385 ± 3.2366 % |

Umbral seleccionado: q = 0.9167 ± 0.0718. Tiempo de test: 0.0956 ± 0.0037 ms/flujo.

## Resultados por semilla

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

## Comparación con el protocolo anterior

Los resultados públicos anteriores de LODA en CIC (F1 ≈ 83,94 %,
FPR ≈ 30,17 %) se obtuvieron con el detector congelado en validación y la
configuración `StandardScaler`, `n_bins=10`, `n_random_cuts=100`. Con el
protocolo online y `Normalizer`, `n_bins=5`, `n_random_cuts=100` el F1 sube
a 90,64 % y el FPR baja a 11,19 % de media, pero la variabilidad entre
semillas es alta: la desviación típica es de 3,24 puntos de F1 y 9,54 de FPR.

## La semilla 8 es un caso atípico

Once de las doce semillas eligen q ≥ 0,90 y quedan en un rango estrecho:

- F1 de test: 89,94-93,40 %
- FPR de test: 3,92-13,10 %

La semilla 8 elige q = 0,70 en validación y se desvía por completo: F1 80,92 %,
FPR 40,24 %. Excluyéndola, el resto de semillas da F1 **91,52 ± 1,11 %** y FPR
**8,55 ± 2,83 %**.

Es el mismo patrón que en NF-UNSW-NB15-v2: el modelo es estable, pero la
selección del cuantil en validación puede escoger un umbral demasiado laxo.
Fijar q en la zona 0,90-0,95 eliminaría buena parte de la dispersión; se deja
como análisis pendiente para no cambiar el protocolo principal.

El barrido también mostró que `Normalizer` domina al resto de scalers en
este dataset: el resto de configuraciones se queda en ~82-85 % de F1 con
~29-36 % de FPR.

## Ficheros

- `raw/`: un JSON por semilla.
- `resumen_multisemilla_loda_cic.json`: resumen agregado completo.
