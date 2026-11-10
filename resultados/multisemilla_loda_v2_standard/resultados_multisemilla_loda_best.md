# Multisemilla de las mejores configuraciones de LODA

Evaluación con 12 semillas de las configuraciones seleccionadas en el
barrido de parámetros. Protocolo online: umbral fijado en validación y
test evaluado una vez.

- Semillas: 0-11
- Flujos de warm-up: 100000

## NF-UNSW-NB15-v2

Configuración: `StandardScaler`, `n_bins=20`, `n_random_cuts=50`.

| Split | Accuracy | Recall | FPR | Precision | F1 |
|---|---:|---:|---:|---:|---:|
| Validación | 97.1398 ± 1.3742 % | 99.9528 ± 0.1634 % | 5.6732 ± 2.6870 % | 94.6839 ± 2.4090 % | 97.2332 ± 1.3005 % |
| Test | 97.6758 ± 1.1332 % | 99.9485 ± 0.1786 % | 4.5969 ± 2.2031 % | 95.6409 ± 2.0149 % | 97.7378 ± 1.0837 % |

Umbral: q = 0.9250 ± 0.0261; valor = 1.130311 ± 0.229951. Tiempo de test: 0.0572 ± 0.0011 ms/flujo.

