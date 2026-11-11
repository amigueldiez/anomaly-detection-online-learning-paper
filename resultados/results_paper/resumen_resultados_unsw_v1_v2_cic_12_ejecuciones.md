# Resumen de resultados: NF-UNSW-NB15, NF-UNSW-NB15-v2 y NF-CSE-CIC-IDS2018

Todos los valores son de **test held-out, media ± desviación típica con n = 12 ejecuciones**.

## 1. Resultados tras las 12 ejecuciones

| Dataset | Modelo | Accuracy | Recall | FPR | Precision | F1 | Parámetro elegido |
|---|---|---:|---:|---:|---:|---:|---|
| **NF-UNSW-NB15** | SAD | 97,6051 ± 0,3651 % | **100,0000 ± 0,0000 %** | 4,7899 ± 0,7302 % | 95,4333 ± 0,6622 % | 97,6622 ± 0,3475 % | umbral 4,8250σ ± 0,1288σ |
| NF-UNSW-NB15 | LODA | 96,7155 ± 0,3863 % | 99,9892 ± 0,0375 % | 6,5582 ± 0,7856 % | 93,8496 ± 0,6959 % | 96,8209 ± 0,3631 % | q 0,9567 ± 0,0156 |
| NF-UNSW-NB15 | **One-Class SVM** | **97,7703 ± 0,1057 %** | 99,5007 ± 0,2249 % | **3,9601 ± 0,0140 %** | **96,1724 ± 0,0053 %** | **97,8081 ± 0,1064 %** | q = 0,99 fijo (ν=0,05, lr=0,1) |
| **NF-UNSW-NB15-v2** | **SAD** | **99,8894 ± 0,0047 %** | **100,0000 ± 0,0000 %** | 0,2213 ± 0,0094 % | 99,7792 ± 0,0094 % | **99,8895 ± 0,0047 %** | umbral 6,2500σ ± 0,7416σ |
| NF-UNSW-NB15-v2 | LODA | 97,6758 ± 1,1332 % | 99,9485 ± 0,1786 % | 4,5969 ± 2,2031 % | 95,6409 ± 2,0149 % | 97,7378 ± 1,0837 % | q 0,9250 ± 0,0261 |
| NF-UNSW-NB15-v2 | One-Class SVM | 99,7514 ± 0,4200 % | 99,5210 ± 0,8463 % | **0,0182 ± 0,0082 %** | **99,9817 ± 0,0082 %** | 99,7491 ± 0,4254 % | q = 0,99 fijo (ν=0,05, lr=0,3) |
| **NF-CSE-CIC-IDS2018** | **SAD** | **91,1956 ± 0,0258 %** | 89,1866 ± 0,0470 % | **6,7954 ± 0,0427 %** | **92,9201 ± 0,0403 %** | **91,0151 ± 0,0269 %** | umbral 3,0σ ± 0,0σ |
| NF-CSE-CIC-IDS2018 | LODA | 90,3442 ± 4,1900 % | **91,8797 ± 1,9985 %** | 11,1913 ± 9,5363 % | 89,7759 ± 6,5839 % | 90,6385 ± 3,2366 % | q 0,9167 ± 0,0718 |
| NF-CSE-CIC-IDS2018 | One-Class SVM | 78,0002 ± 0,0524 % | 69,7872 ± 0,1731 % | 13,7869 ± 0,1735 % | 83,5038 ± 0,1468 % | 76,0316 ± 0,0714 % | q = 0,9 fijo (ν=0,5, lr=0,3) |

En negrita, el mejor valor de cada columna dentro de cada dataset.

> **Aviso (2026-11-11).** Las filas de SAD proceden del protocolo anterior y se
> consideran inválidas; sus artefactos se eliminaron. Se mantienen aquí solo como
> referencia histórica. Las filas de LODA se han actualizado al protocolo online
> con las configuraciones finales de cada dataset. Los resultados de One-Class
> SVM siguen vigentes.

## 2. Parámetros por par modelo-dataset

| Dataset | Modelo | Scaler | Modelo / hiperparámetros | Umbral / selección | Semillas |
|---|---|---|---|---|---|
| NF-UNSW-NB15 | SAD | `MaxAbsScaler` | Un `StandardAbsoluteDeviation` por característica, `sub_stat=Mean`, `var=Var`, agregación `max`, filtro de varianza `> 1e-12` | Rejilla `0,0–10,0σ` paso `0,1` + `12/15/20` (104 candidatos); F1 en validación; empates → mediana del plateau | `0–11` |
| NF-UNSW-NB15-v2 | SAD | `MaxAbsScaler` | Ídem | Ídem | `0–11` |
| NF-CSE-CIC-IDS2018 | SAD | `MaxAbsScaler` | Ídem | Rejilla `0,0–10,0σ` paso `0,5` + `12/15/20` (24 candidatos); F1 en validación | `0–11` |
| NF-UNSW-NB15 | LODA | `Normalizer` | `LODA(n_bins=5, n_random_cuts=100, seed=s)` | Cuantiles de los scores de entrenamiento: `0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99`; F1 en validación; reentreno online con flujos no marcados | `0–11` |
| NF-UNSW-NB15-v2 | LODA | `StandardScaler` | `LODA(n_bins=20, n_random_cuts=50, seed=s)` | Ídem | `0–11` |
| NF-CSE-CIC-IDS2018 | LODA | `Normalizer` | `LODA(n_bins=5, n_random_cuts=100, seed=s)` | Ídem | `0–11` |
| NF-UNSW-NB15 | One-Class SVM | `MaxAbsScaler` | `QuantileFilter(OneClassSVM(nu=0.05, intercept_lr=InverseScaling(0.1)), q=0.99)` | Una sola configuración; F1 en validación | 12 ejecuciones independientes, **sin semilla fija** |
| NF-UNSW-NB15-v2 | One-Class SVM | `MaxAbsScaler` | `QuantileFilter(OneClassSVM(nu=0.05, intercept_lr=InverseScaling(0.3)), q=0.99)` | Una sola configuración; F1 en validación | 12 ejecuciones independientes, **sin semilla fija** |
| NF-CSE-CIC-IDS2018 | One-Class SVM | `StandardScaler` | `QuantileFilter(OneClassSVM(nu=0.5, intercept_lr=InverseScaling(0.3)), q=0.9)` | Configuración fija (mejor de la rejilla de 48); F1 en validación | `0–11` |

## 3. Parámetros comunes a todos los pares

| Parámetro | Valor |
|---|---|
| `FLOWS_TRAIN_SCALER` | `1000` |
| `FLOWS_TRAIN_OML` | `100000` |
| `VALIDATION_RATIO` | `0.5` |
| `SELECTION_METRIC` | `f1` |
| Features | `IPV4_SRC_ADDR, IPV4_DST_ADDR, L4_DST_PORT, L4_SRC_PORT, PROTOCOL, IN_BYTES, OUT_BYTES, FLOW_DURATION_MILLISECONDS, Label` |
| Validación / test | Balanceados al 50 % |
| Partición UNSW | Posicional sobre el CSV original (bloques consecutivos) |
| Partición CIC | CSV pre-barajado con `random.seed(42)` |

## 4. Comandos para reproducir

**SAD**

```bash
# UNSW v1
.venv/bin/python sad_detector.py --filename ../NF-UNSW-NB15.csv \
  --seed <s> --aggregation max --threshold-step 0.1 --output-json salida.json

# UNSW v2
.venv/bin/python sad_detector.py --filename ../NF-UNSW-NB15-v2.csv \
  --seed <s> --aggregation max --threshold-step 0.1 --output-json salida.json

# CIC-IDS2018
.venv/bin/python sad_detector.py \
  --filename /tmp/opencode/NF-CSE-CIC-IDS2018-shuffled.csv \
  --seed <s> --aggregation max --threshold-step 0.5 --output-json salida.json
```

**LODA**

```bash
# UNSW v1
.venv/bin/python loda_detector.py --filename ../NF-UNSW-NB15.csv \
  --seed <s> --scaling-type Normalizer --n-bins 5 --n-random-cuts 100 \
  --output-json salida.json

# UNSW v2
.venv/bin/python loda_detector.py --filename ../NF-UNSW-NB15-v2.csv \
  --seed <s> --scaling-type StandardScaler --n-bins 20 --n-random-cuts 50 \
  --output-json salida.json

# CIC-IDS2018
.venv/bin/python loda_detector.py \
  --filename /tmp/opencode/NF-CSE-CIC-IDS2018-shuffled.csv \
  --seed <s> --scaling-type Normalizer --n-bins 5 --n-random-cuts 100 \
  --output-json salida.json
```

**One-Class SVM**

```bash
# UNSW v1 y v2: configurar config.yaml y lanzar 12 veces sin semilla
# v1: FILENAME ../NF-UNSW-NB15.csv,    MaxAbsScaler, nu 0.05, q 0.99, lr 0.1
# v2: FILENAME ../NF-UNSW-NB15-v2.csv, MaxAbsScaler, nu 0.05, q 0.99, lr 0.3
.venv/bin/python multiexecution.py

# CIC-IDS2018
.venv/bin/python ocsvm_cic.py \
  --filename /tmp/opencode/NF-CSE-CIC-IDS2018-shuffled.csv \
  --seed <s> --scaling-type StandardScaler --nu 0.5 --q 0.9 --learning-rate 0.3 \
  --output-json salida.json
```

**Nota importante:** en UNSW, SAD y LODA usan semillas `0–11`; el SVM de UNSW son 12 ejecuciones independientes **sin semilla fija** (cada una baraja el orden aleatoriamente), tal como se documentó en `resultados_split_entrenamiento_validacion_test.md`. En CIC los tres modelos usan semillas `0–11` sobre el mismo CSV barajado.
