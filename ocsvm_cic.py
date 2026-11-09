#!/usr/bin/env python3
"""Fixed-hyperparameter One-Class SVM evaluation on NF-CSE-CIC-IDS2018.

This script mirrors the single-configuration path of
``hyperparameter_optimization.py``:

1. Build the same disjoint scaler / training / validation / test partitions.
2. Fit the scaler and train the model on benign flows only.
3. Evaluate the validation split online (the model updates itself with the
   flows it does not flag).
4. Evaluate the held-out test split once with the same model.

The hyperparameters are fixed to the best configuration found by the grid in
``resultados_ocsvm_cic_ids2018.md`` (StandardScaler, nu=0.5, q=0.9, lr=0.3).
The only thing that changes between executions is the random seed, which
controls the internal order of the four sets (the partition membership itself
is fixed by the pre-shuffled CSV).
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

import hyperparameter_optimization as hp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--filename',
                        default='/tmp/opencode/NF-CSE-CIC-IDS2018-shuffled.csv',
                        help='Path to the pre-shuffled labeled CSV (default: %(default)s)')
    parser.add_argument('--flows-train-oml', type=int, default=hp.FLOWS_TRAIN_OML,
                        help='Number of benign flows used for training (default: %(default)s)')
    parser.add_argument('--scaling-type', default='StandardScaler',
                        choices=['MaxAbsScaler', 'MinMaxScaler', 'Normalizer',
                                 'RobustScaler', 'StandardScaler'],
                        help='Scaler fitted on the benign warm-up flows (default: %(default)s)')
    parser.add_argument('--nu', type=float, default=0.5,
                        help='nu parameter of the One-Class SVM (default: %(default)s)')
    parser.add_argument('--q', type=float, default=0.9,
                        help='q parameter of the QuantileFilter (default: %(default)s)')
    parser.add_argument('--learning-rate', type=float, default=0.3,
                        help='intercept learning rate (default: %(default)s)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for the internal order of the sets (default: %(default)s)')
    parser.add_argument('--output-json', default=None,
                        help='Optional path to save the metrics as JSON (default: disabled)')
    args = parser.parse_args()

    np.random.seed(args.seed)

    print(f'Dataset: {args.filename}')
    print(f'Warm-up flows: {args.flows_train_oml} | scaler: {args.scaling_type} '
          f'| nu: {args.nu} | q: {args.q} | lr: {args.learning_rate} '
          f'| seed: {args.seed}')

    parameter = hp.Parameters(
        hp.FEATURES, args.scaling_type, hp.FLOWS_TRAIN_SCALER,
        args.nu, args.q, args.learning_rate, args.flows_train_oml, 0, 0, 0,
    )

    splits = hp.generate_dataset(
        args.filename,
        flows_train_oml=args.flows_train_oml,
        save_logs=False,
    )
    scaled = hp.scale_datasets(parameter, splits)

    print('\nTraining One-Class SVM...')
    start = time.time()
    model, training_time = hp.train_model_OML(scaled['train'], parameter)
    print(f'Training time: {training_time:.2f} s')

    print('\nEvaluating validation split (online)...')
    start = time.time()
    val_metrics, _ = hp.evaluate_stream(
        model, scaled['validation'], splits['validation_labeled'],
    )
    validation_time = time.time() - start
    print(f'Validation time: {validation_time:.2f} s')

    print('\nEvaluating held-out test split (online)...')
    start = time.time()
    test_metrics, _ = hp.evaluate_stream(
        model, scaled['test'], splits['test_labeled'],
    )
    test_time = time.time() - start
    print(f'Test time: {test_time:.2f} s')

    payload = {
        'dataset': args.filename,
        'model': 'OneClassSVM+QuantileFilter',
        'scaling_type': args.scaling_type,
        'nu': args.nu,
        'q': args.q,
        'learning_rate': args.learning_rate,
        'seed': args.seed,
        'flows_train_oml': args.flows_train_oml,
        'selected_q': args.q,
        'validation': val_metrics,
        'test': test_metrics,
        'timings': {
            'training_seconds': training_time,
            'validation_seconds': validation_time,
            'test_seconds': test_time,
            'test_ms_per_flow': 1000 * test_time / len(scaled['test']),
        },
    }

    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2) + '\n')
        print(f'\nMetrics saved to {output}')


if __name__ == '__main__':
    main()
