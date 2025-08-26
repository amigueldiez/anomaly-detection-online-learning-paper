# Anomaly detection in network flows using unsupervised online machine learning

> Alberto Miguel-Diez, Adrián Campazas-Vega, Ángel Manuel Guerrero-Higueras, Claudia Álvarez-Aparicio and Vicente Matellán-Olivera

## Abstract

Nowadays, the volume of network traffic continues to grow, along with the frequency and sophistication of attacks. This scenario highlights the need for solutions capable of continuously adapting, since network behavior is dynamic and changes over time. This work presents an anomaly detection model for network flows using unsupervised machine learning with online learning capabilities. This approach allows the system to dynamically learn the normal behavior of the network and detect deviations without requiring labeled data, which is particularly useful in real-world environments where traffic is constantly changing and labeled data is scarce. The model was implemented using the River library with a One-Class SVM and evaluated on the NF-UNSW-NB15 dataset and its extended version v2, which contain network flows labeled with different attack categories. The results show an accuracy above 98%, a false positive rate below 3.1%, and a recall of 100% in the most advanced version of the dataset. In addition, the low processing time per flow (<0.033 ms) demonstrates the feasibility of the approach for real-time applications.

## Files

This repository contains the following files and folders:

* **imgs/**: Directory containing images used in the project documentation (e.g., acknowledgements logos).

* **.gitignore**: Specifies files and directories that should be ignored by Git (e.g., temporary files, virtual environments).

* **CITATION.cff**: Citation File Format metadata that allows others to cite this software correctly in academic works.

* **README.md**: Main documentation file providing an overview of the project.

* **config.yaml**: Configuration file containing the main parameters and dataset paths used by the scripts.

* **hyperparameter\_optimization.py**: Script to optimize hyperparameters of the One-Class SVM model using different scalers and learning rate schedulers.

* **multiexecution.py**: Script to automate multiple runs of the hyperparameter optimization process.

* **parameters.py**: Class Parameter.

## Usage

To run the hyperparameter optimization script, use the following command:

```bash
python3 hyperparameter_optimization.py
```

To run with multiple executions, use the following command (the attribute `NUM_EXECUTIONS` controls the number of runs):

```bash
python3 multiexecution.py
```

It's important to update the `config.yaml` file with the appropriate dataset paths and parameters before running the scripts.

## Acknowledgements

This research is part of the CIBERLAB project, financed "by European Union NextGeneration-EU, the Recovery Plan, Transformation and Resilience, through INCIBE".

![logo of INCIBE](imgs/INCIBE_logo.jpg)