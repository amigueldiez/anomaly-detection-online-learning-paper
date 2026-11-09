# Modelo One-Class SVM sobre NF-CSE-CIC-IDS2018 barajado

Prueba del pipeline principal (`hyperparameter_optimization.py`, One-Class SVM de
River con `QuantileFilter`) sobre NF-CSE-CIC-IDS2018 v1, usando el mismo esquema
de particiones y el mismo protocolo de selección en validación que en
NF-UNSW-NB15.

El fichero se baraja antes de partirlo porque CIC-IDS2018 está ordenado por día
de captura y tipo de ataque, y el split posicional del pipeline calibraría el
umbral sobre una mezcla de ataques distinta de la que se usa en test.

## Metadatos

| Campo | Valor |
|---|---|
| Fecha de ejecución | 2026-11-06 |
| Rama | `feature/more-models` |
| Commit | `474f363` |
| Script | `hyperparameter_optimization.py` |
| Ejecuciones | 1 configuración base + 48 configuraciones de rejilla |
| Semilla del barajado | `42` (`random.seed(42)` sobre el CSV completo) |
| Semilla del split interno | No fijada; `split_flows` baraja cada conjunto con `.sample(frac=1)` |
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

Particiones resultantes:

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

## Por qué se baraja el fichero

`split_flows` construye validación y test con **bloques consecutivos** del CSV:
los benignos de validación son las filas siguientes a las de entrenamiento y los
de test, el bloque siguiente; las anomalías se reparten por orden de aparición.
Eso solo es válido si bloques consecutivos son intercambiables.

En CIC-IDS2018 no lo son. Composición por bloques del fichero original (20
bloques de ~419.619 filas):

| Bloque | Benigno | Ataques dominantes |
|---|---:|---|
| 0-14 | 96-100 % | casi nada (solo Infilteration) |
| 15 | 71,7 % | SlowHTTPTest 24 %, Hulk 3 % |
| 16 | 46,4 % | LOIC-HTTP 30 %, Hulk 22 % |
| 17 | 43,7 % | LOIC-HTTP 56 % |
| 18 | 63,6 % | FTP-BruteForce 19 %, GoldenEye 7 % |
| 19 | 50,6 % | FTP-BruteForce 26 %, SSH-Bruteforce 22 % |

Con el split posicional, la mitad de anomalías que va a validación y la que va a
test son familias distintas:

| Validación (orden de fichero) | | Test (orden de fichero) | |
|---|---:|---|---:|
| DDoS LOIC-HTTP | 41,9 % | FTP-BruteForce | 37,9 % |
| DoS Hulk | 21,2 % | DDoS LOIC-HTTP | 32,3 % |
| DoS SlowHTTPTest | 20,7 % | SSH-Bruteforce | 18,5 % |
| Infilteration | 12,2 % | DoS GoldenEye | 6,4 % |
| Bot | 3,1 % | DoS Slowloris | 4,5 % |

Por eso se baraja el fichero completo con `random.seed(42)` antes de aplicar el
pipeline: así validación y test son muestras i.i.d. del mismo dataset y el umbral
se selecciona sobre la misma distribución que se evalúa.

## Configuración

Parámetros comunes:

| Parámetro | Valor |
|---|---|
| `FILENAME` | `/tmp/opencode/NF-CSE-CIC-IDS2018-shuffled.csv` |
| `TYPE_IP` | `IPv4` |
| `SIGNIFICANT_PORT` | `False` |
| `BENIGN_FLOWS_ADDITIONAL` | `0` |
| `FLOWS_TRAIN_SCALER` | `1.000` |
| `FLOWS_TRAIN_OML` | `100.000` |
| `VALIDATION_RATIO` | `0.5` |
| `SELECTION_METRIC` | `f1` |
| Modelo | `QuantileFilter(OneClassSVM(nu, intercept_lr=InverseScaling(lr)), q)` |
| Selección | mejor F1 sobre validación; test evaluado una sola vez |

### Configuración base (la del artículo para UNSW v1)

`MaxAbsScaler`, ν = 0.05, q = 0.99, lr = 0.1.

### Rejilla de hiperparámetros

48 combinaciones:

| Parámetro | Valores |
|---|---|
| `scaling_type` | `MaxAbsScaler`, `StandardScaler` |
| `nu_value` | 0.05, 0.2, 0.5 |
| `q_value` | 0.5, 0.7, 0.9, 0.99 |
| `learning_rate` | 0.1, 0.3 |

## Resultados

### Configuración base del artículo

| Conjunto | Accuracy | Recall | FPR | Precision | F1 | TP / TN / FP / FN |
|---|---:|---:|---:|---:|---:|---:|
| Validación | 38,4633 % | 2,3036 % | 25,3771 % | 8,3220 % | 3,6083 % | 11.739 / 380.280 / 129.322 / 497.863 |
| **Test (held-out)** | **38,4085 %** | **2,3903 %** | **25,5733 %** | **8,5479 %** | **3,7359 %** | 12.181 / 379.279 / 130.322 / 497.420 |

Tiempos: entrenamiento 0,55 s; test 5,58 s (0,0055 ms/flujo).

Con los hiperparámetros que funcionan en UNSW, el SVM **no detecta
prácticamente nada** en CIC y aun así marca un 25 % de benignos como anómalos.

### Rejilla de hiperparámetros (validación)

Top 10 configuraciones por F1 de validación:

| Scaler | ν | q | lr | Accuracy | Recall | FPR | Precision | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| StandardScaler | 0.5 | 0.9 | 0.3 | 78,0882 % | 69,7566 % | 13,5802 % | 83,7044 % | **76,0967 %** |
| StandardScaler | 0.5 | 0.9 | 0.1 | 77,9908 % | 69,6218 % | 13,6403 % | 83,6177 % | 75,9806 % |
| StandardScaler | 0.5 | 0.7 | 0.3 | 73,1225 % | 83,6333 % | 37,3884 % | 69,1060 % | 75,6788 % |
| StandardScaler | 0.5 | 0.7 | 0.1 | 73,0028 % | 83,3525 % | 37,3470 % | 69,0579 % | 75,5348 % |
| StandardScaler | 0.2 | 0.7 | 0.3 | 71,5083 % | 79,2128 % | 36,1963 % | 68,6365 % | 73,5464 % |
| StandardScaler | 0.2 | 0.7 | 0.1 | 71,4213 % | 78,8900 % | 36,0473 % | 68,6374 % | 73,4074 % |
| StandardScaler | 0.05 | 0.7 | 0.3 | 70,2814 % | 77,0866 % | 36,5238 % | 67,8517 % | 72,1750 % |
| StandardScaler | 0.05 | 0.7 | 0.1 | 68,0461 % | 74,7101 % | 38,6178 % | 65,9238 % | 70,0425 % |
| StandardScaler | 0.2 | 0.5 | 0.1 | 55,3691 % | 95,9290 % | 85,1908 % | 52,9644 % | 68,2478 % |
| StandardScaler | 0.5 | 0.5 | 0.3 | 54,4164 % | 97,4586 % | 88,6258 % | 52,3733 % | 68,1328 % |

Mejor configuración por escalador:

| Scaler | ν | q | lr | Accuracy | Recall | FPR | Precision | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MaxAbsScaler | 0.05 | 0.5 | 0.1 | 49,6007 % | 98,3526 % | 99,1513 % | 49,7978 % | 66,1185 % |
| **StandardScaler** | **0.5** | **0.9** | **0.3** | **78,0882 %** | **69,7566 %** | **13,5802 %** | **83,7044 %** | **76,0967 %** |

`MaxAbsScaler` es inviable en CIC: su mejor configuración marca el **99,15 %**
de los benignos como anómalos (F1 66,12 %). La causa es la disparidad de escalas
del vector de características, porque el escalador divide cada columna por su
máximo absoluto:

| Característica | Valor máximo observado |
|---|---:|
| IP (entero) | 4.294.967.295 |
| OUT_BYTES | 214.122.854 |
| IN_BYTES | 24.371.270 |
| FLOW_DURATION_MILLISECONDS | 4.294.967 |
| Puertos | 65.535 |
| PROTOCOL | 58 |

Tras `MaxAbsScaler`, las IPs ocupan el rango 0–1 mientras que los bytes y la
duración típicos quedan en el orden de 10⁻⁶. El hiperplano del SVM queda
dominado por las IPs y las características de tráfico pierden peso.

### Configuración seleccionada y test held-out

Configuración elegida en validación por F1: **`StandardScaler`, ν = 0.5,
q = 0.9, lr = 0.3**.

| Conjunto | Accuracy | Recall | FPR | Precision | F1 | TP / TN / FP / FN |
|---|---:|---:|---:|---:|---:|---:|
| Validación | 78,0882 % | 69,7566 % | 13,5802 % | 83,7044 % | 76,0967 % | 355.481 / 440.397 / 69.205 / 154.121 |
| **Test (held-out)** | **78,0588 %** | **69,6272 %** | **13,5096 %** | **83,7502 %** | **76,0385 %** | 354.821 / 440.756 / 68.845 / 154.780 |

Tiempos: entrenamiento 0,50 s; test 4,53 s (**0,0044 ms/flujo**).

Validación y test coinciden (F1 76,0967 % vs 76,0385 %), así que el resultado es
estable para este split.

### Mejor configuración con FPR ≤ 3,1 %

Si se impone el objetivo de FPR del artículo (< 3,1 %), lo mejor que da el SVM
es:

| Scaler | ν | q | lr | Accuracy | Recall | FPR | Precision | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| StandardScaler | 0.5 | 0.99 | 0.1 | 69,3486 % | 41,7112 % | 3,0139 % | 93,2612 % | 57,6419 % |

Es decir: para bajar el FPR al objetivo hay que renunciar a más de la mitad del
recall (41,71 %). No hay un punto de operación con recall alto y FPR bajo.

## Comparación con SAD, LODA y UNSW

Todos los valores de CIC corresponden al mismo fichero barajado (semilla 42) y
al mismo esquema de particiones.

| Dataset | Modelo | Accuracy | Recall | FPR | Precision | F1 |
|---|---|---:|---:|---:|---:|---:|
| NF-CSE-CIC-IDS2018 | **SAD** (MaxAbs, 100 cortes) | 91,1695 % | 89,1384 % | 6,7994 % | 92,9127 % | **90,9864 %** |
| NF-CSE-CIC-IDS2018 | LODA (Standard, 100 cortes) | 81,9831 % | 94,1362 % | 30,1701 % | 75,7292 % | 83,9354 % |
| NF-CSE-CIC-IDS2018 | **One-Class SVM** (Standard, ν=0.5, q=0.9, lr=0.3) | 78,0588 % | 69,6272 % | 13,5096 % | 83,7502 % | **76,0385 %** |
| NF-UNSW-NB15 | One-Class SVM | 97,7703 % | 99,5007 % | 3,9601 % | 96,1724 % | 97,8081 % |
| NF-UNSW-NB15-v2 | One-Class SVM | 99,7514 % | 99,5210 % | 0,0182 % | 99,9817 % | 99,7491 % |

Los valores de SAD y LODA proceden de `resultados_sad_cic_ids2018.md` y
`resultados_loda_cic_ids2018.md`; los de UNSW, de
`resultados_split_entrenamiento_validacion_test.md` (media de 12 ejecuciones).
La comparación es orientativa: en UNSW el split es el posicional del pipeline,
mientras que en CIC el fichero está barajado.

## Observaciones

1. **El One-Class SVM es el peor de los tres modelos en CIC.** Con la mejor
   configuración de la rejilla obtiene F1 76,04 %, frente a 83,94 % de LODA y
   90,99 % de SAD. El buen resultado en UNSW no se transfiere a CIC.
2. **Los hiperparámetros del artículo no son transferibles.** La configuración
   base de UNSW v1 (MaxAbs, ν=0.05, q=0.99, lr=0.1) da F1 3,74 % en CIC. Hay que
   reajustar por dataset; si no, la comparación es engañosa.
3. **`MaxAbsScaler` es inviable en CIC.** Su mejor configuración tiene FPR
   99,15 %. La disparidad de escalas (IPs del orden de 10⁹ frente a bytes
   típicos del orden de 10²) hace que, tras dividir cada columna por su máximo,
   el hiperplano quede dominado por las IPs y las características de tráfico
   pierdan peso. En UNSW el mismo escalador funciona bien, así que el problema
   es la combinación de escalado y heterogeneidad del benigno en CIC, no el
   escalador por sí solo.
4. **El modelo de River es lineal.** `OneClassSVM` no usa kernel, así que solo
   puede trazar un hiperplano. El tráfico benigno de CIC es mucho más
   heterogéneo (servicios, días y capturas distintas) y un hiperplano no lo
   separa. SAD, que evalúa cada característica por separado, se adapta mejor.
5. **No hay punto de operación bueno.** La configuración que maximiza F1 tiene
   un FPR del 13,51 %. Para cumplir el objetivo de FPR < 3,1 % hay que aceptar un
   recall del 41,71 % (F1 57,64 %).
6. **La selección por F1 sobre un conjunto balanceado 50/50 favorece
   configuraciones con FPR alto.** Con q bajo el modelo marca casi todo y el F1
   sube artificialmente. Conviene reportar también el punto con FPR restringido.
7. **Una sola ejecución.** No se reporta media ± desviación típica. Además, el
   orden interno de los conjuntos no se fijó (`.sample(frac=1)` sin semilla),
   como en `resultados_split_entrenamiento_validacion_test.md`, así que los
   valores pueden variar ligeramente entre ejecuciones.
8. **Validación y test están balanceados al 50 %.** El FPR y el recall deben
   interpretarse con esa proporción; en tráfico real, mucho más desbalanceado,
   la precisión sería menor.

## Reproducibilidad

Crear el fichero barajado:

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

Configuración base:

```yaml
FILENAME: '/tmp/opencode/NF-CSE-CIC-IDS2018-shuffled.csv'
scaling_type: ['MaxAbsScaler']
nu_value: [0.05]
q_value: [0.99]
learning_rate: [0.1]
FLOWS_TRAIN_SCALER: 1000
FLOWS_TRAIN_OML: 100000
VALIDATION_RATIO: 0.5
SELECTION_METRIC: 'f1'
```

Rejilla:

```yaml
scaling_type: ['MaxAbsScaler', 'StandardScaler']
nu_value: [0.05, 0.2, 0.5]
q_value: [0.5, 0.7, 0.9, 0.99]
learning_rate: [0.1, 0.3]
```

Ejecutar desde un directorio que contenga el `config.yaml` (el módulo lo lee de
`config.yaml` en el directorio de trabajo):

```bash
cd /tmp/opencode/run_cic_ocsvm
MPLBACKEND=Agg PYTHONPATH="/home/alberto/Online machine learning/codigo-paper-oml-computer-security" \
  "/home/alberto/Online machine learning/codigo-paper-oml-computer-security/.venv/bin/python" \
  -u -c "import hyperparameter_optimization as hp; hp.main()"
```

Para que la ejecución sea completamente determinista, añadir
`import numpy as np; np.random.seed(42)` antes de `hp.main()`; el resto de la
partición ya queda fijado por el CSV barajado.

Artefactos de esta ejecución: `/tmp/opencode/run_cic_ocsvm/run.log` (configuración
base), `/tmp/opencode/run_cic_ocsvm/grid.log` (rejilla) y los plots generados en
ese mismo directorio.

## Repeticiones multisemilla en CIC

Los resultados de este documento corresponden a una sola ejecución. La
evaluación con 12 semillas sobre el mismo CSV barajado (partición fija, solo
cambia el orden interno de los conjuntos) y con la configuración fija
`StandardScaler`, ν = 0.5, q = 0.9, lr = 0.3 está en
`resultados_multisemilla_cic_ids2018.md`. Resumen de test (media ± desviación
típica, n = 12):

| Dataset | Accuracy | Recall | FPR | F1 |
|---|---:|---:|---:|---:|
| NF-CSE-CIC-IDS2018 | 78,0002 ± 0,0524 % | 69,7872 ± 0,1731 % | 13,7869 ± 0,1735 % | 76,0316 ± 0,0714 % |

La desviación es muy pequeña: la ejecución individual (F1 76,04 %) es
representativa del rendimiento medio con esta configuración.
