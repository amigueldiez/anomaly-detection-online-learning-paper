"""Helpers to summarize and aggregate wall-clock timings.

Latencies are measured per flow with :func:`time.perf_counter_ns` and stored in
nanoseconds.  ``summarize_latencies_ns`` turns those samples into the
distribution statistics reported in the paper (mean and standard deviation,
median, 95th/99th percentiles and maximum), while ``aggregate_timings``
combines the per-run timing blocks of the multi-seed executions into
mean +/- sample standard deviation.
"""

import math


def _percentile(sorted_values, percentile):
    """Linear-interpolated percentile (same rule as ``numpy.percentile``)."""
    n = len(sorted_values)
    if n == 1:
        return sorted_values[0]
    rank = (n - 1) * percentile / 100.0
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return sorted_values[lower]
    weight = rank - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _mean_std(values):
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    mean = sum(values) / n
    if n < 2:
        return mean, 0.0
    variance = sum((value - mean) ** 2 for value in values) / (n - 1)
    return mean, math.sqrt(variance)


def summarize_latencies_ns(samples_ns):
    """Return the latency distribution (in milliseconds) of ``samples_ns``.

    ``std_ms`` is the sample standard deviation (``n - 1`` denominator), so it
    can be compared with the dispersion reported for the rest of the metrics.
    """
    values_ms = sorted(sample / 1_000_000 for sample in samples_ns)
    if not values_ms:
        return {
            'count': 0,
            'mean_ms': 0.0,
            'std_ms': 0.0,
            'min_ms': 0.0,
            'median_ms': 0.0,
            'p95_ms': 0.0,
            'p99_ms': 0.0,
            'max_ms': 0.0,
        }

    mean, std = _mean_std(values_ms)
    return {
        'count': len(values_ms),
        'mean_ms': mean,
        'std_ms': std,
        'min_ms': values_ms[0],
        'median_ms': _percentile(values_ms, 50.0),
        'p95_ms': _percentile(values_ms, 95.0),
        'p99_ms': _percentile(values_ms, 99.0),
        'max_ms': values_ms[-1],
    }


def _aggregate_block(runs, key):
    values = [run['timings'][key] for run in runs]
    first = values[0]
    if isinstance(first, dict):
        return {
            sub_key: dict(zip(('mean', 'std'), _mean_std([
                value[sub_key] for value in values
            ])))
            for sub_key in first
        }
    return dict(zip(('mean', 'std'), _mean_std(values)))


def aggregate_timings(runs, keys=None):
    """Aggregate per-run timing blocks into mean +/- sample standard deviation.

    Scalar entries become ``{'mean': ..., 'std': ...}`` and nested blocks (such
    as the per-flow ``latency_ms`` distribution) are aggregated key by key.
    """
    if not runs:
        return {}
    if keys is None:
        keys = sorted(runs[0]['timings'])
    return {key: _aggregate_block(runs, key) for key in keys}


def format_aggregated_timings(timings):
    """Render an aggregated timing block as human-readable lines."""
    lines = []
    for key in sorted(timings):
        block = timings[key]
        if isinstance(block, dict) and 'mean' in block:
            lines.append(f'    {key:40s} {block["mean"]:12.4f} ± {block["std"]:.4f}')
        elif isinstance(block, dict):
            for sub_key in sorted(block):
                stats = block[sub_key]
                lines.append(
                    f'    {key + "/" + sub_key:40s} '
                    f'{stats["mean"]:12.6f} ± {stats["std"]:.6f}'
                )
    return lines
