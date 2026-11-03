#!/usr/bin/env python3
"""Multi-seed evaluation of SAD and LODA on the two NF-UNSW-NB15 datasets.

The protocol mirrors ``multiexecution.py``: the membership of each flow to the
scaler / training / validation / test sets is fixed, and every execution only
changes the internal order of those sets (plus LODA's random projections,
because the same seed is passed to the model). Therefore the reported standard
deviation measures the effect of the online learning order, not the variability
of the partition itself.

For every model, dataset and seed the corresponding script is run as a separate
process with ``--output-json``. The per-run JSON files and an aggregated summary
(mean + sample standard deviation) are written under ``resultados/``.
"""

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

NUM_EXECUTIONS = 12
SEED_START = 0

DATASETS = {
    'NF-UNSW-NB15': '../NF-UNSW-NB15.csv',
    'NF-UNSW-NB15-v2': '../NF-UNSW-NB15-v2.csv',
}

MODELS = {
    'SAD': {
        'script': 'sad_unsw.py',
        'args': ['--aggregation', 'max', '--threshold-step', '0.1'],
        'selected_key': 'selected_threshold',
        'configuration': 'MaxAbsScaler, aggregation=max, threshold grid 0.0-10.0 sigma step 0.1',
    },
    'LODA': {
        'script': 'loda_unsw.py',
        'args': ['--scaling-type', 'StandardScaler', '--n-bins', '10', '--n-random-cuts', '100'],
        'selected_key': 'selected_q',
        'configuration': 'StandardScaler, n_bins=10, n_random_cuts=100',
    },
}

METRICS = ('accuracy', 'recall', 'false_positive', 'precision', 'f1')


def mean_std(values):
    """Return the mean and the sample standard deviation of ``values``."""
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    mean = sum(values) / n
    if n < 2:
        return mean, 0.0
    variance = sum((value - mean) ** 2 for value in values) / (n - 1)
    return mean, math.sqrt(variance)


def aggregate_metrics(runs, split):
    """Mean and standard deviation of every metric over the given runs."""
    return {
        metric: dict(zip(('mean', 'std'), mean_std([run[split][metric] for run in runs])))
        for metric in METRICS
    }


def aggregate_timings(runs, keys):
    return {
        key: dict(zip(('mean', 'std'), mean_std([run['timings'][key] for run in runs])))
        for key in keys
    }


def run_one(model, dataset, seed, filename, raw_path, output_dir):
    config = MODELS[model]
    command = [
        sys.executable,
        config['script'],
        '--filename', filename,
        '--seed', str(seed),
        '--output-json', str(raw_path),
    ] + config['args']

    print(f'\n=== {model} | {dataset} | seed {seed} ===', flush=True)
    print(' '.join(command), flush=True)
    start = time.time()
    result = subprocess.run(command, capture_output=True, text=True)
    elapsed = time.time() - start

    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(
            f'{model} on {dataset} with seed {seed} failed '
            f'(return code {result.returncode})'
        )

    payload = json.loads(raw_path.read_text())
    payload['run_seconds'] = elapsed
    return payload


def print_summary(summary):
    print('\n' + '=' * 78)
    print(f"Results over {summary['protocol']['num_executions']} executions (mean ± std)")
    print('=' * 78)
    for model, datasets in summary['results'].items():
        for dataset, entry in datasets.items():
            print(f'\n{model} | {dataset}')
            print(f"  Selected parameter ({entry['selected_key']}): "
                  f"{entry['selected_parameter']['mean']:.4f} "
                  f"± {entry['selected_parameter']['std']:.4f}")
            for split in ('validation', 'test'):
                print(f'  {split.capitalize()}:')
                for metric, stats in entry[split].items():
                    print(f'    {metric:15s} {100 * stats["mean"]:9.4f} % '
                          f'± {100 * stats["std"]:.4f} %')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--num-executions', type=int, default=NUM_EXECUTIONS,
                        help='Number of seeds per model and dataset (default: %(default)s)')
    parser.add_argument('--seed-start', type=int, default=SEED_START,
                        help='First seed; consecutive integers are used (default: %(default)s)')
    parser.add_argument('--models', nargs='+', default=list(MODELS), choices=list(MODELS),
                        help='Models to run (default: all)')
    parser.add_argument('--datasets', nargs='+', default=list(DATASETS), choices=list(DATASETS),
                        help='Datasets to run (default: all)')
    parser.add_argument('--output-dir', default='resultados',
                        help='Directory for the JSON outputs (default: %(default)s)')
    parser.add_argument('--skip-existing', action='store_true',
                        help='Reuse raw JSON files that already exist')
    args = parser.parse_args()

    seeds = list(range(args.seed_start, args.seed_start + args.num_executions))
    output_dir = Path(args.output_dir)
    raw_dir = output_dir / 'multisemilla_sad_loda' / 'raw'
    raw_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        'protocol': {
            'num_executions': args.num_executions,
            'seeds': seeds,
            'split': (
                'Fixed partition membership (as in multiexecution.py): the seed only '
                'changes the order inside the scaler/training/validation/test sets. '
                'For LODA the same seed also controls the random projections.'
            ),
            'datasets': {name: path for name, path in DATASETS.items() if name in args.datasets},
            'models': {
                name: MODELS[name]['configuration']
                for name in args.models
            },
        },
        'results': {},
    }

    started = time.time()
    for model in args.models:
        summary['results'][model] = {}
        for dataset in args.datasets:
            runs = []
            for seed in seeds:
                safe_dataset = dataset.replace(' ', '_')
                raw_path = raw_dir / f'{model}_{safe_dataset}_seed{seed}.json'
                if args.skip_existing and raw_path.exists():
                    payload = json.loads(raw_path.read_text())
                else:
                    payload = run_one(
                        model, dataset, seed, DATASETS[dataset], raw_path, output_dir,
                    )
                payload['seed'] = seed
                runs.append(payload)

            selected_values = [run[selected] for run in runs
                               for selected in [MODELS[model]['selected_key']]]
            timing_keys = sorted(runs[0]['timings'])
            summary['results'][model][dataset] = {
                'script': MODELS[model]['script'],
                'configuration': MODELS[model]['configuration'],
                'selected_key': MODELS[model]['selected_key'],
                'selected_parameter': dict(zip(
                    ('mean', 'std'), mean_std(selected_values),
                )),
                'validation': aggregate_metrics(runs, 'validation'),
                'test': aggregate_metrics(runs, 'test'),
                'timings': aggregate_timings(runs, timing_keys),
                'runs': [
                    {
                        'seed': run['seed'],
                        'selected_parameter': run[MODELS[model]['selected_key']],
                        'validation': run['validation'],
                        'test': run['test'],
                        'timings': run['timings'],
                        'run_seconds': run['run_seconds'],
                    }
                    for run in runs
                ],
            }
            print(f'\nFinished {model} | {dataset} '
                  f'({len(runs)} runs, {time.time() - started:.1f} s elapsed)', flush=True)

    summary['total_seconds'] = time.time() - started
    summary_path = output_dir / 'multisemilla_sad_loda_unsw.json'
    summary_path.write_text(json.dumps(summary, indent=2) + '\n')
    print_summary(summary)
    print(f'\nSummary saved to {summary_path}')


if __name__ == '__main__':
    main()
