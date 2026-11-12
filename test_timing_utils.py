import os
import sys
import unittest

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(REPO_ROOT)
sys.path.insert(0, REPO_ROOT)

import timing_utils as tu


class TestSummarizeLatencies(unittest.TestCase):
    def test_empty_sample_returns_zeroed_summary(self):
        stats = tu.summarize_latencies_ns([])

        self.assertEqual(stats['count'], 0)
        for key in ('mean_ms', 'std_ms', 'min_ms', 'median_ms',
                    'p95_ms', 'p99_ms', 'max_ms'):
            self.assertEqual(stats[key], 0.0)

    def test_single_sample_has_no_dispersion(self):
        stats = tu.summarize_latencies_ns([5_000_000])

        self.assertEqual(stats['count'], 1)
        self.assertEqual(stats['std_ms'], 0.0)
        for key in ('mean_ms', 'min_ms', 'median_ms', 'p95_ms', 'p99_ms', 'max_ms'):
            self.assertEqual(stats[key], 5.0)

    def test_mean_std_and_max_of_known_sample(self):
        stats = tu.summarize_latencies_ns(
            [1_000_000, 2_000_000, 3_000_000, 4_000_000]
        )

        self.assertEqual(stats['count'], 4)
        self.assertAlmostEqual(stats['mean_ms'], 2.5)
        self.assertAlmostEqual(stats['std_ms'], (5.0 / 3.0) ** 0.5)
        self.assertAlmostEqual(stats['min_ms'], 1.0)
        self.assertAlmostEqual(stats['median_ms'], 2.5)
        self.assertAlmostEqual(stats['max_ms'], 4.0)

    def test_percentiles_use_linear_interpolation(self):
        stats = tu.summarize_latencies_ns(
            [1_000_000, 2_000_000, 3_000_000, 4_000_000]
        )

        self.assertAlmostEqual(stats['p95_ms'], 3.85)
        self.assertAlmostEqual(stats['p99_ms'], 3.97)

    def test_is_order_independent(self):
        stats = tu.summarize_latencies_ns(
            [4_000_000, 1_000_000, 3_000_000, 2_000_000]
        )

        self.assertAlmostEqual(stats['median_ms'], 2.5)
        self.assertAlmostEqual(stats['max_ms'], 4.0)


class TestAggregateTimings(unittest.TestCase):
    def test_aggregates_scalar_keys_across_runs(self):
        runs = [
            {'timings': {'training_seconds': 1.0, 'evaluation_seconds': 10.0}},
            {'timings': {'training_seconds': 3.0, 'evaluation_seconds': 14.0}},
        ]

        summary = tu.aggregate_timings(runs)

        self.assertAlmostEqual(summary['training_seconds']['mean'], 2.0)
        self.assertAlmostEqual(summary['training_seconds']['std'], 2.0 ** 0.5)
        self.assertAlmostEqual(summary['evaluation_seconds']['mean'], 12.0)

    def test_aggregates_nested_latency_statistics(self):
        runs = [
            {'timings': {'latency_ms': {'mean_ms': 2.0, 'max_ms': 5.0}}},
            {'timings': {'latency_ms': {'mean_ms': 4.0, 'max_ms': 9.0}}},
        ]

        summary = tu.aggregate_timings(runs)

        self.assertAlmostEqual(summary['latency_ms']['mean_ms']['mean'], 3.0)
        self.assertAlmostEqual(
            summary['latency_ms']['mean_ms']['std'], 2.0 ** 0.5
        )
        self.assertAlmostEqual(summary['latency_ms']['max_ms']['mean'], 7.0)
        self.assertAlmostEqual(summary['latency_ms']['max_ms']['std'], 8.0 ** 0.5)

    def test_can_restrict_to_a_subset_of_keys(self):
        runs = [
            {'timings': {'a': 1.0, 'b': 2.0}},
            {'timings': {'a': 3.0, 'b': 4.0}},
        ]

        summary = tu.aggregate_timings(runs, keys=['b'])

        self.assertEqual(sorted(summary), ['b'])
        self.assertAlmostEqual(summary['b']['mean'], 3.0)

    def test_format_lists_training_evaluation_and_latency(self):
        timings = {
            'total_training_seconds': {'mean': 12.5, 'std': 0.5},
            'evaluation_seconds': {'mean': 5.0, 'std': 0.1},
            'evaluation_latency_ms': {
                'mean_ms': {'mean': 0.07, 'std': 0.001},
                'p95_ms': {'mean': 0.09, 'std': 0.002},
                'max_ms': {'mean': 0.20, 'std': 0.010},
            },
        }

        text = '\n'.join(tu.format_aggregated_timings(timings))

        self.assertIn('total_training_seconds', text)
        self.assertIn('evaluation_seconds', text)
        self.assertIn('evaluation_latency_ms/mean_ms', text)
        self.assertIn('evaluation_latency_ms/p95_ms', text)
        self.assertIn('evaluation_latency_ms/max_ms', text)


if __name__ == '__main__':
    unittest.main()
