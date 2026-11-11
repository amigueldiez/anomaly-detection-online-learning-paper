# Multisemilla de las mejores configuraciones de LODA

Evaluación con 12 semillas de las configuraciones seleccionadas en el
barrido de parámetros. Protocolo online: umbral fijado en validación y
test evaluado una vez.

- Semillas: 0-11
- Flujos de warm-up: 100000

## NF-UNSW-NB15

Configuración: `Normalizer`, `n_bins=5`, `n_random_cuts=100`.

| Split | Accuracy | Recall | FPR | Precision | F1 |
|---|---:|---:|---:|---:|---:|
| Validación | 97.8649 ± 0.3569 % | 99.9949 ± 0.0175 % | 4.2651 ± 0.7234 % | 95.9135 ± 0.6674 % | 97.9106 ± 0.3425 % |
| Test | 96.7155 ± 0.3863 % | 99.9892 ± 0.0375 % | 6.5582 ± 0.7856 % | 93.8496 ± 0.6959 % | 96.8209 ± 0.3631 % |

Umbral: q = 0.9567 ± 0.0156; valor = -9.839889 ± 1.019877. Tiempo de test: 0.0924 ± 0.0017 ms/flujo.

