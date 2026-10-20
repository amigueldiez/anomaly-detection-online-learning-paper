import os
import sys
import tempfile
import unittest

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
# The module under test reads config.yaml relative to the working directory.
os.chdir(REPO_ROOT)
sys.path.insert(0, REPO_ROOT)

import pandas as pd

import hyperparameter_optimization as hpo
import multiexecution as me


def make_labeled_df(n_benign, n_anomalous):
    """Build a tiny labeled flow dataset (benign first, then anomalous)."""
    return pd.DataFrame(
        {
            "value": list(range(n_benign + n_anomalous)),
            "Label": [0] * n_benign + [1] * n_anomalous,
        }
    )


class TestSplitFlows(unittest.TestCase):
    def setUp(self):
        self.df = make_labeled_df(100, 40)

    def test_returns_expected_keys(self):
        splits = hpo.split_flows(self.df, 10, 20, validation_ratio=0.5)
        self.assertEqual(set(splits), {"scaler", "train", "validation", "test"})

    def test_scaler_and_train_sizes(self):
        splits = hpo.split_flows(self.df, 10, 20, validation_ratio=0.5)
        self.assertEqual(len(splits["scaler"]), 10)
        self.assertEqual(len(splits["train"]), 20)

    def test_scaler_and_train_are_only_benign(self):
        splits = hpo.split_flows(self.df, 10, 20, validation_ratio=0.5)
        self.assertTrue((splits["scaler"]["Label"] == 0).all())
        self.assertTrue((splits["train"]["Label"] == 0).all())

    def test_validation_and_test_are_balanced(self):
        splits = hpo.split_flows(self.df, 10, 20, validation_ratio=0.5)
        for name in ("validation", "test"):
            counts = splits[name]["Label"].value_counts()
            self.assertEqual(counts.get(0, 0), counts.get(1, 0))

    def test_splits_are_pairwise_disjoint(self):
        splits = hpo.split_flows(self.df, 10, 20, validation_ratio=0.5)
        index_sets = [set(s.index) for s in splits.values()]
        for a in range(len(index_sets)):
            for b in range(a + 1, len(index_sets)):
                self.assertEqual(index_sets[a] & index_sets[b], set())

    def test_validation_ratio_controls_anomaly_split(self):
        splits = hpo.split_flows(self.df, 10, 20, validation_ratio=0.25)
        # 25% of the 40 anomalies -> 10 in validation, 30 in test (balanced).
        self.assertEqual(len(splits["validation"]), 10 + 10)
        self.assertEqual(len(splits["test"]), 30 + 30)

    def test_raises_when_not_enough_benign(self):
        with self.assertRaises(ValueError):
            hpo.split_flows(make_labeled_df(5, 10), 10, 20, validation_ratio=0.5)


class TestClassificationMetrics(unittest.TestCase):
    def test_confusion_counts_and_derived_metrics(self):
        labels = [1, 1, 1, 0, 0, 0]
        predictions = [True, True, False, False, True, False]
        metrics = hpo.classification_metrics(labels, predictions)
        self.assertEqual(metrics["tp"], 2)
        self.assertEqual(metrics["fn"], 1)
        self.assertEqual(metrics["tn"], 2)
        self.assertEqual(metrics["fp"], 1)
        self.assertAlmostEqual(metrics["accuracy"], 4 / 6)
        self.assertAlmostEqual(metrics["precision"], 2 / 3)
        self.assertAlmostEqual(metrics["recall"], 2 / 3)
        self.assertAlmostEqual(metrics["false_positive"], 1 / 3)
        self.assertAlmostEqual(metrics["f1"], 2 / 3)


class TestSelectBestParameter(unittest.TestCase):
    def setUp(self):
        self.p1, self.p2, self.p3 = object(), object(), object()

    def test_selects_highest_f1(self):
        results = [
            (self.p1, {"f1": 0.5}),
            (self.p2, {"f1": 0.9}),
            (self.p3, {"f1": 0.7}),
        ]
        self.assertIs(hpo.select_best_parameter(results, "f1"), self.p2)

    def test_selects_lowest_false_positive(self):
        results = [
            (self.p1, {"false_positive": 0.1}),
            (self.p2, {"false_positive": 0.3}),
            (self.p3, {"false_positive": 0.2}),
        ]
        self.assertIs(hpo.select_best_parameter(results, "false_positive"), self.p1)

    def test_raises_on_empty_results(self):
        with self.assertRaises(ValueError):
            hpo.select_best_parameter([], "f1")


def write_synthetic_csv(path, n_benign=60, n_anomalous=20):
    import numpy as np

    rows = []
    for i in range(n_benign + n_anomalous):
        label = 0 if i < n_benign else 1
        rows.append({
            "IPV4_SRC_ADDR": f"10.0.{i // 256}.{i % 256}",
            "IPV4_DST_ADDR": f"10.1.{i // 256}.{i % 256}",
            "L4_DST_PORT": 80 + (i % 5),
            "L4_SRC_PORT": 1024 + (i % 7),
            "PROTOCOL": 6,
            "IN_BYTES": 100 + i,
            "OUT_BYTES": 200 + i * 2,
            "FLOW_DURATION_MILLISECONDS": 10 + (i % 3),
            "Label": label,
            "Attack": "Benign" if label == 0 else "Exploits",
        })
    pd.DataFrame(rows).to_csv(path, index=False)


class TestGenerateDatasetIntegration(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.csv_path = os.path.join(self.tmp.name, "synthetic.csv")
        write_synthetic_csv(self.csv_path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_splits_are_disjoint_and_well_formed(self):
        splits = hpo.generate_dataset(
            self.csv_path,
            flows_train_scaler=10,
            flows_train_oml=20,
            validation_ratio=0.5,
            benign_additional=0,
            save_logs=False,
        )

        self.assertEqual(set(splits), {
            "scaler", "train", "validation", "test",
            "validation_labeled", "test_labeled",
        })
        self.assertEqual(len(splits["scaler"]), 10)
        self.assertEqual(len(splits["train"]), 20)

        # The scaler set has no label column and the evaluation sets are balanced.
        self.assertNotIn("Label", splits["scaler"].columns)
        self.assertTrue((splits["validation_labeled"]["Label"].value_counts().get(0, 0)
                         == splits["validation_labeled"]["Label"].value_counts().get(1, 0)))
        self.assertTrue((splits["test_labeled"]["Label"].value_counts().get(0, 0)
                         == splits["test_labeled"]["Label"].value_counts().get(1, 0)))

        # The four sets must be pairwise disjoint. ``IN_BYTES`` is unique per
        # synthetic flow, so it acts as a stable row identifier.
        in_bytes_index = hpo.FEATURES.index("IN_BYTES")
        row_ids = [
            set(splits["scaler"]["IN_BYTES"]),
            set(splits["train"][:, in_bytes_index]),
            set(splits["validation_labeled"]["IN_BYTES"]),
            set(splits["test_labeled"]["IN_BYTES"]),
        ]
        for a in range(len(row_ids)):
            for b in range(a + 1, len(row_ids)):
                self.assertEqual(row_ids[a] & row_ids[b], set())

    def test_scaler_does_not_learn_from_validation_or_test(self):
        # MaxAbsScaler divides by the largest absolute value seen so far. The
        # validation value (100) is far above anything in scaler/train (max 10),
        # so a held-out scaler must produce a scaled value > 1.
        splits = {
            "scaler": pd.DataFrame([{0: 1.0, 1: 1.0}]),
            "train": [[10.0, 1.0]],
            "validation": [[100.0, 1.0]],
            "test": [[1.0, 1.0]],
        }
        parameter = hpo.Parameters(hpo.FEATURES, "MaxAbsScaler", 1, 0.05, 0.99, 0.1, 1, 0, 0, 0)
        scaled = hpo.scale_datasets(parameter, splits)
        self.assertGreater(scaled["validation"][0][0], 1.0)

    def test_scaling_and_evaluation_end_to_end(self):
        splits = hpo.generate_dataset(
            self.csv_path,
            flows_train_scaler=10,
            flows_train_oml=20,
            validation_ratio=0.5,
            benign_additional=0,
            save_logs=False,
        )
        parameter = hpo.Parameters(hpo.FEATURES, "MaxAbsScaler", 10, 0.05, 0.99, 0.1, 20, 0, 0, 0)

        scaled = hpo.scale_datasets(parameter, splits)
        self.assertEqual(len(scaled["train"]), len(splits["train"]))
        self.assertEqual(len(scaled["validation"]), len(splits["validation"]))
        self.assertEqual(len(scaled["test"]), len(splits["test"]))

        model, _ = hpo.train_model_OML(scaled["train"], parameter)
        metrics, curves = hpo.evaluate_stream(model, scaled["validation"], splits["validation_labeled"])

        for key in ("tp", "tn", "fp", "fn", "accuracy", "precision", "recall", "false_positive", "f1"):
            self.assertIn(key, metrics)
        self.assertEqual(len(curves["accuracy"]), len(scaled["validation"]))


class TestMeanStd(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(me.mean_std([]), (0.0, 0.0))

    def test_single_value(self):
        self.assertEqual(me.mean_std([5.0]), (5.0, 0.0))

    def test_mean_and_sample_std(self):
        mean, std = me.mean_std([1.0, 2.0, 3.0, 4.0])
        self.assertAlmostEqual(mean, 2.5)
        self.assertAlmostEqual(std, (5.0 / 3.0) ** 0.5)

    def test_two_values(self):
        mean, std = me.mean_std([2.0, 4.0])
        self.assertAlmostEqual(mean, 3.0)
        self.assertAlmostEqual(std, 2.0 ** 0.5)


if __name__ == "__main__":
    unittest.main()
