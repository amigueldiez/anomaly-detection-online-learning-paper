import os
import sys
import unittest

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
# The module under test reads config.yaml relative to the working directory.
os.chdir(REPO_ROOT)
sys.path.insert(0, REPO_ROOT)

import sad_detector as sd


class RecordingModel:
    """Minimal stand-in that records which rows are learned."""

    def __init__(self, scores):
        self._scores = iter(scores)
        self.learned = []

    def score_one(self, x):
        return next(self._scores)

    def learn_one(self, x):
        self.learned.append(x)


class TestEvaluateStream(unittest.TestCase):
    def test_updates_model_only_above_the_threshold(self):
        model = RecordingModel([5.0, 1.0, 4.0])
        rows = [[1.0], [2.0], [3.0]]
        labels = [1, 0, 1]

        sd.evaluate_stream(model, rows, labels, threshold=3.0)

        self.assertEqual(model.learned, [[2.0]])

    def test_collects_one_latency_sample_per_flow(self):
        model = RecordingModel([5.0, 1.0, 5.0, 1.0])
        rows = [[0.0], [0.0], [0.0], [0.0]]
        labels = [1, 0, 0, 1]

        metrics, scores, latencies_ns = sd.evaluate_stream(
            model, rows, labels, threshold=3.0, collect_latency=True,
        )

        self.assertEqual(len(scores), len(rows))
        self.assertEqual(len(latencies_ns), len(rows))
        self.assertTrue(all(sample > 0 for sample in latencies_ns))
        self.assertEqual(metrics['tp'], 1)
        self.assertEqual(metrics['tn'], 1)


if __name__ == '__main__':
    unittest.main()
