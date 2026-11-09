import os
import sys
import unittest

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
# The module under test reads config.yaml relative to the working directory.
os.chdir(REPO_ROOT)
sys.path.insert(0, REPO_ROOT)

import loda_detector as ld


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
    def test_updates_model_only_on_non_anomalous_flows(self):
        model = RecordingModel([5.0, 1.0, 4.0])
        rows = [[1.0], [2.0], [3.0]]
        labels = [1, 0, 1]

        ld.evaluate_stream(model, rows, labels, threshold=3.0)

        self.assertEqual(model.learned, [{'feature_0': 2.0}])

    def test_threshold_is_inclusive(self):
        model = RecordingModel([3.0])

        ld.evaluate_stream(model, [[1.0]], [0], threshold=3.0)

        self.assertEqual(model.learned, [])

    def test_returns_metrics_from_labels(self):
        model = RecordingModel([5.0, 1.0, 5.0, 1.0])
        rows = [[0.0], [0.0], [0.0], [0.0]]
        labels = [1, 0, 0, 1]

        metrics = ld.evaluate_stream(model, rows, labels, threshold=3.0)

        self.assertEqual(metrics['tp'], 1)
        self.assertEqual(metrics['tn'], 1)
        self.assertEqual(metrics['fp'], 1)
        self.assertEqual(metrics['fn'], 1)


if __name__ == '__main__':
    unittest.main()
