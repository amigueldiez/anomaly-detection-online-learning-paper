#!/usr/bin/env python3
"""Functional test of River's StandardAbsoluteDeviation (SAD) on NF-UNSW-NB15.

SAD is a univariate detector: ``learn_one(x, y)`` / ``score_one(x, y)`` only
look at the scalar ``y``. To process multivariate network flows, one SAD
instance is kept per feature and the per-flow score is an aggregation (max by
default) of the individual z-scores. Features whose variance is still zero
after the warm-up are ignored, because their z-score would explode.

Protocol (no leakage):
1. Fit the scaler and train the SAD ensemble on benign flows only.
2. Evaluate the validation set online for a grid of thresholds and keep the one
   that optimises ``SELECTION_METRIC`` (F1 by default). The exact same online
   protocol (score, classify, learn when not anomalous) is used here and on
   the test set.
3. Retrain the ensemble from scratch and evaluate the test set once, using the
   threshold selected in step 2.
"""

import argparse
import time

import numpy as np

import hyperparameter_optimization as hp
from river import anomaly


class SADEnsemble:
    """One :class:`river.anomaly.StandardAbsoluteDeviation` per feature."""

    def __init__(self, n_features, aggregation='max'):
        self.detectors = [anomaly.StandardAbsoluteDeviation() for _ in range(n_features)]
        self.aggregation = aggregation

    def learn_one(self, row):
        for detector, value in zip(self.detectors, row):
            detector.learn_one(None, value)

    def score_one(self, row):
        scores = [
            detector.score_one(None, value)
            for detector, value in zip(self.detectors, row)
            if detector.variance.get() > 1e-12
        ]
        if not scores:
            return 0.0
        if self.aggregation == 'mean':
            return float(np.mean(scores))
        if self.aggregation == 'median':
            return float(np.median(scores))
        return float(np.max(scores))


def train_model(n_features, aggregation, train_rows):
    model = SADEnsemble(n_features, aggregation=aggregation)
    for row in train_rows:
        model.learn_one(row)
    return model


def evaluate_stream(model, dataset_no_labels, labels, threshold):
    """Online evaluation: score, classify, and update when not anomalous."""
    predictions = []
    scores = []
    for row, label in zip(dataset_no_labels, labels):
        score = model.score_one(row)
        anomalous = score > threshold
        predictions.append(anomalous)
        scores.append(score)
        if not anomalous:
            model.learn_one(row)
    return hp.classification_metrics(labels, predictions), scores


def candidate_thresholds(upper=10.0, step=0.1):
    """Threshold grid used for the validation search."""
    grid = np.round(np.arange(0.0, upper + 1e-9, step), 1)
    return np.concatenate((grid, [12.0, 15.0, 20.0]))


def select_threshold(train_rows, validation_rows, labels, n_features, aggregation,
                     metric='f1'):
    """Select the threshold on validation using the same online protocol as test.

    For every candidate threshold a fresh model is trained on the benign
    warm-up flows and evaluated online over the validation stream (the model
    updates itself with the flows it does not flag). This mirrors exactly what
    happens on the test set, so the selected threshold is not biased by a
    frozen-vs-online mismatch. The test set is never touched here.

    Ties are broken by taking the median threshold of the tied plateau, so the
    decision boundary is not placed at either edge.
    """
    results = []
    for threshold in candidate_thresholds():
        model = train_model(n_features, aggregation, train_rows)
        metrics, _ = evaluate_stream(model, validation_rows, labels, threshold)
        results.append((float(threshold), metrics))

    if metric in hp.LOWER_IS_BETTER:
        best_value = min(metrics[metric] for _, metrics in results)
    else:
        best_value = max(metrics[metric] for _, metrics in results)

    tied = [result for result in results if result[1][metric] == best_value]
    # With a 0.1 grid the best metric usually holds over a plateau of
    # thresholds. Taking the median of that plateau keeps the boundary away
    # from both edges, which is more robust than picking the highest one
    # (right on top of the lowest anomaly score) or the lowest one.
    best = min(tied, key=lambda result: abs(result[0] - np.median([t for t, _ in tied])))

    return best[0], best[1], results


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
    parser.add_argument('--flows-train-oml', type=int, default=hp.FLOWS_TRAIN_OML,
                        help='Number of benign flows used for the warm-up (default: %(default)s)')
    parser.add_argument('--aggregation', choices=['max', 'mean', 'median'], default='max',
                        help='How per-feature scores are combined (default: %(default)s)')
    parser.add_argument('--selection-metric', default=hp.SELECTION_METRIC,
                        choices=['accuracy', 'precision', 'recall', 'false_positive', 'f1'],
                        help='Metric used to select the threshold on validation '
                             '(default: %(default)s)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for the dataset split (default: %(default)s)')
    parser.add_argument('--filename', default=hp.FILENAME,
                        help='Path to the labeled CSV dataset (default: %(default)s)')
    args = parser.parse_args()

    np.random.seed(args.seed)

    print(f'Dataset: {args.filename}')
    print(f'Warm-up flows: {args.flows_train_oml} | aggregation: {args.aggregation} '
          f'| selection metric: {args.selection_metric} | seed: {args.seed}')

    parameter = hp.Parameters(
        hp.FEATURES, 'MaxAbsScaler', hp.FLOWS_TRAIN_SCALER,
        0, 0, 0, args.flows_train_oml, 0, 0, 0,
    )

    splits = hp.generate_dataset(args.filename, flows_train_oml=args.flows_train_oml,
                                 save_logs=False)
    scaled = hp.scale_datasets(parameter, splits)

    n_features = len(hp.FEATURES) - 1
    val_labels = list(splits['validation_labeled']['Label'])
    test_labels = list(splits['test_labeled']['Label'])

    # --- Phase 1: select the threshold on validation (test is never touched) ---
    print('\nSelecting the threshold on the validation set...')
    start = time.time()
    threshold, val_metrics, results = select_threshold(
        scaled['train'], scaled['validation'], val_labels,
        n_features, args.aggregation, args.selection_metric,
    )
    selection_time = time.time() - start
    print(f'Threshold search time: {selection_time:.2f} s '
          f'({len(results)} candidates, model retrained for each one)')
    print(f'Selected threshold on validation: {threshold:.1f} sigma '
          f'(metric: {args.selection_metric})')
    tied = [t for t, metrics in results
            if metrics[args.selection_metric] == val_metrics[args.selection_metric]]
    print(f'Tied best thresholds: {min(tied):.1f}-{max(tied):.1f} sigma '
          f'({len(tied)} candidates); median selected')
    print_metrics('Validation (threshold selected here)', val_metrics)

    top = sorted(results, key=lambda result: result[1][args.selection_metric],
                 reverse=args.selection_metric not in hp.LOWER_IS_BETTER)[:5]
    print('\nTop 5 thresholds on validation:')
    print(f"  {'thr':>6} {'accuracy':>9} {'recall':>8} {'FPR':>9} "
          f"{'precision':>10} {'F1':>8}")
    for candidate, metrics in top:
        print(f"  {candidate:6.1f} {metrics['accuracy']:9.6f} {metrics['recall']:8.6f} "
              f"{metrics['false_positive']:9.6f} {metrics['precision']:10.6f} "
              f"{metrics['f1']:8.6f}")

    # --- Phase 2: retrain from scratch and evaluate the held-out test set ---
    print('\nRetraining from scratch for the final evaluation...')
    start = time.time()
    model = train_model(n_features, args.aggregation, scaled['train'])
    print(f'Training time: {time.time() - start:.2f} s')

    start = time.time()
    test_metrics, _ = evaluate_stream(model, scaled['test'], test_labels, threshold)
    test_time = time.time() - start
    print_metrics('Test (held out, threshold fixed on validation)', test_metrics, test_time)

    print(f'\nFeatures used: {n_features}')


if __name__ == '__main__':
    main()
