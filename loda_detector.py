#!/usr/bin/env python3
"""Train River's LODA on benign flows and evaluate it on a labeled dataset.

The script is dataset-agnostic: it works with any CSV that contains the columns
listed in ``FEATURES`` in ``config.yaml`` (the first two must be valid IPv4
addresses) and a binary ``Label`` column (0 = benign, 1 = anomalous). Use
``--filename`` to select the dataset; the default is the path configured in
``config.yaml`` (currently NF-UNSW-NB15).

LODA (Lightweight On-line Detector of Anomalies) is an ensemble of one-dimensional
histograms built on sparse random projections. It is trained here only with benign
flows; the anomaly threshold is then derived from the distribution of the training
scores (quantiles) and selected on the validation split.

This is deliberately different from ``hyperparameter_optimization.py``, which wraps
the detector in ``anomaly.QuantileFilter`` from the very first sample. That wrapper
can freeze LODA during warm-up: LODA assigns its maximum score to values outside its
histograms, and with ``protect_anomaly_detector=True`` the filter stops updating the
detector after the first out-of-range score. Training LODA directly avoids that
feedback loop.

Protocol (no leakage):
1. Fit the scaler and train LODA on benign flows only.
2. Compute LODA scores for those benign flows and derive candidate thresholds as
   quantiles of the training scores.
3. Evaluate every candidate on the validation split (frozen detector) and keep the
   one that optimises ``SELECTION_METRIC`` (F1 by default).
4. Evaluate the held-out test split once with the selected threshold.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from river import anomaly

import hyperparameter_optimization as hp


SCALERS = ('MaxAbsScaler', 'MinMaxScaler', 'Normalizer', 'RobustScaler', 'StandardScaler')


def as_dict(row):
    """Convert a scaled row (list of values) to the dict LODA expects."""
    return {f'feature_{i}': value for i, value in enumerate(row)}


def train_loda(train_rows, n_bins, n_random_cuts, seed):
    """Train LODA directly on the benign warm-up rows."""
    model = anomaly.LODA(n_bins=n_bins, n_random_cuts=n_random_cuts, seed=seed)
    for row in train_rows:
        model.learn_one(as_dict(row))
    return model


def score_stream(model, rows):
    """Score a stream without updating the model."""
    return np.fromiter(
        (model.score_one(as_dict(row)) for row in rows),
        dtype=float,
        count=len(rows),
    )


def evaluate_scores(scores, labels, threshold):
    """Classify pre-computed scores and return the repository's metric dict."""
    predictions = scores >= threshold
    return hp.classification_metrics(labels, predictions)


def parse_q_grid(text):
    quantiles = [float(value.strip()) for value in text.split(',') if value.strip()]
    if not quantiles:
        raise ValueError('The quantile grid cannot be empty')
    if any(not 0.0 < q < 1.0 for q in quantiles):
        raise ValueError('Quantiles must be strictly between 0 and 1')
    return quantiles


def select_quantile(results, metric):
    """Select the best quantile, breaking ties at the median of the tied range."""
    if metric in hp.LOWER_IS_BETTER:
        best_value = min(result['metrics'][metric] for result in results)
    else:
        best_value = max(result['metrics'][metric] for result in results)

    tied = [result for result in results if result['metrics'][metric] == best_value]
    tied = sorted(tied, key=lambda result: result['q'])
    return tied[len(tied) // 2], tied


def print_metrics(title, metrics, elapsed=None):
    print(f'--- {title} ---')
    if elapsed is not None:
        print(f'Time: {elapsed:.2f} s')
    print(f"Accuracy:            {metrics['accuracy']:.6f}")
    print(f"Recall:              {metrics['recall']:.6f}")
    print(f"False Positive Rate: {metrics['false_positive']:.6f}")
    print(f"Precision:           {metrics['precision']:.6f}")
    print(f"F1 Score:            {metrics['f1']:.6f}")
    print(f"TP: {metrics['tp']}  TN: {metrics['tn']}  "
          f"FP: {metrics['fp']}  FN: {metrics['fn']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--filename', default=hp.FILENAME,
                        help='Path to the labeled CSV dataset (default: %(default)s)')
    parser.add_argument('--flows-train-oml', type=int, default=hp.FLOWS_TRAIN_OML,
                        help='Number of benign flows used to train LODA (default: %(default)s)')
    parser.add_argument('--scaling-type', default='StandardScaler', choices=SCALERS,
                        help='Scaler fitted on the benign warm-up flows (default: %(default)s)')
    parser.add_argument('--n-bins', type=int, default=10,
                        help='Maximum number of bins per histogram (default: %(default)s)')
    parser.add_argument('--n-random-cuts', type=int, default=100,
                        help='Number of random projections (default: %(default)s)')
    parser.add_argument('--q-grid', default='0.5,0.6,0.7,0.8,0.85,0.9,0.95,0.99',
                        help='Comma-separated quantiles of the training scores to test '
                             '(default: %(default)s)')
    parser.add_argument('--selection-metric', default=hp.SELECTION_METRIC,
                        choices=['accuracy', 'precision', 'recall', 'false_positive', 'f1'],
                        help='Metric used to select the threshold on validation '
                             '(default: %(default)s)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for the split and for LODA (default: %(default)s)')
    parser.add_argument('--output-json', default=None,
                        help='Optional path to save the metrics as JSON (default: disabled)')
    args = parser.parse_args()

    np.random.seed(args.seed)
    quantiles = parse_q_grid(args.q_grid)

    print(f'Dataset: {args.filename}')
    print(f'Warm-up flows: {args.flows_train_oml} | scaler: {args.scaling_type} '
          f'| bins: {args.n_bins} | cuts: {args.n_random_cuts} '
          f'| selection metric: {args.selection_metric} | seed: {args.seed}')

    parameter = hp.Parameters(
        hp.FEATURES, args.scaling_type, hp.FLOWS_TRAIN_SCALER,
        0, 0, 0, args.flows_train_oml, 0, 0, 0,
    )

    splits = hp.generate_dataset(
        args.filename,
        flows_train_oml=args.flows_train_oml,
        save_logs=False,
    )
    scaled = hp.scale_datasets(parameter, splits)

    val_labels = list(splits['validation_labeled']['Label'])
    test_labels = list(splits['test_labeled']['Label'])

    # --- Phase 1: train LODA on benign flows and score the training stream ---
    print('\nTraining LODA on benign flows...')
    start = time.time()
    model = train_loda(scaled['train'], args.n_bins, args.n_random_cuts, args.seed)
    training_time = time.time() - start
    print(f'Training time: {training_time:.2f} s')

    start = time.time()
    train_scores = score_stream(model, scaled['train'])
    train_score_time = time.time() - start

    start = time.time()
    val_scores = score_stream(model, scaled['validation'])
    val_score_time = time.time() - start
    print(f'Training-score time: {train_score_time:.2f} s '
          f'| validation-score time: {val_score_time:.2f} s')

    # --- Phase 2: select the threshold on validation (test is never touched) ---
    print('\nSelecting the threshold on the validation set...')
    start = time.time()
    results = []
    for q in quantiles:
        threshold = float(np.quantile(train_scores, q))
        metrics = evaluate_scores(val_scores, val_labels, threshold)
        results.append({'q': q, 'threshold': threshold, 'metrics': metrics})
    selection_time = time.time() - start

    best, tied = select_quantile(results, args.selection_metric)
    print(f'Threshold search time: {selection_time:.2f} s ({len(results)} candidates)')
    print(f"Selected q: {best['q']:.4f} "
          f"(threshold {best['threshold']:.6f}; metric: {args.selection_metric})")
    if len(tied) > 1:
        print(f"Tied q values: {tied[0]['q']:.4f}-{tied[-1]['q']:.4f} "
              f"({len(tied)} candidates); median selected")
    print_metrics('Validation (threshold selected here)', best['metrics'])

    top = sorted(
        results,
        key=lambda result: result['metrics'][args.selection_metric],
        reverse=args.selection_metric not in hp.LOWER_IS_BETTER,
    )[:5]
    print('\nTop 5 quantiles on validation:')
    print(f"  {'q':>6} {'threshold':>12} {'accuracy':>9} {'recall':>8} "
          f"{'FPR':>9} {'precision':>10} {'F1':>8}")
    for result in top:
        metrics = result['metrics']
        print(f"  {result['q']:6.4f} {result['threshold']:12.6f} "
              f"{metrics['accuracy']:9.6f} {metrics['recall']:8.6f} "
              f"{metrics['false_positive']:9.6f} {metrics['precision']:10.6f} "
              f"{metrics['f1']:8.6f}")

    # --- Phase 3: evaluate the held-out test set once ---
    print('\nEvaluating the held-out test set...')
    start = time.time()
    test_scores = score_stream(model, scaled['test'])
    test_score_time = time.time() - start
    test_metrics = evaluate_scores(test_scores, test_labels, best['threshold'])
    print_metrics('Test (held out, threshold fixed on validation)', test_metrics,
                  test_score_time)

    payload = {
        'dataset': args.filename,
        'scaling_type': args.scaling_type,
        'n_bins': args.n_bins,
        'n_random_cuts': args.n_random_cuts,
        'seed': args.seed,
        'flows_train_oml': args.flows_train_oml,
        'selection_metric': args.selection_metric,
        'quantile_grid': quantiles,
        'selected_q': best['q'],
        'threshold': best['threshold'],
        'validation': best['metrics'],
        'test': test_metrics,
        'validation_top5': top,
        'timings': {
            'training_seconds': training_time,
            'train_score_seconds': train_score_time,
            'validation_score_seconds': val_score_time,
            'threshold_selection_seconds': selection_time,
            'test_score_seconds': test_score_time,
            'test_ms_per_flow': 1000 * test_score_time / len(scaled['test']),
        },
    }
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2) + '\n')
        print(f'\nMetrics saved to {output}')


if __name__ == '__main__':
    main()
